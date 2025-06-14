# Core Python imports
import asyncio
import os  
import logging  
from enum import Enum  
from typing import ClassVar, List, Any  

# Third-party library imports
from pymavlink import mavutil  
from pydantic import BaseModel, Field  
import re 
import requests 
from bs4 import BeautifulSoup  
import types 
import json  
from datetime import datetime, timedelta 

# Semantic Kernel components for orchestrating AI workflows
from semantic_kernel import Kernel
from semantic_kernel.agents import OpenAIResponsesAgent
from semantic_kernel.connectors.ai.open_ai import OpenAISettings
from semantic_kernel.functions import kernel_function
from semantic_kernel.processes import ProcessBuilder
from semantic_kernel.processes.kernel_process import (
    KernelProcess,
    KernelProcessEvent,
    KernelProcessEventVisibility,
    KernelProcessStep,
    KernelProcessStepContext,
    KernelProcessStepState,
)
from semantic_kernel.processes.local_runtime.local_kernel_process import start as start_local_process

# Define chatbot events for inter-step communication within a KernelProcess
class ChatbotEvents(Enum):
    StartProcess = "StartProcess"
    SimpleResponse = "SimpleResponse"
    ComplexResponse = "ComplexResponse"
    ExecutionSuccess = "ExecutionSuccess"
    ExecutionFail = "ExecutionFail"
    PlanResponse = "PlanResponse"

# OpenAI client setup
from openai import OpenAI
client = OpenAI()

# Step to determine if a query is simple or complex
class DeciderStep(KernelProcessStep):
    DecideQuery: ClassVar[str] = "decide_query"
    
    @kernel_function(name=DecideQuery)
    async def decide_query(self, input, context: KernelProcessStepContext):
        prompt = """
        You have to evaluate if the user query is simple or complex. Simple means you have information about that user question. Complex means the query is realted to the flight logger you will need UAV file to provide the correct answer.
        Just respond with simple or complex.
        """
        # Call GPT to classify query
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps(input["query"])},
            ],
        ).choices[0].message.content

        # Emit appropriate event based on classification
        if "simple" in response.lower():
            await context.emit_event(ChatbotEvents.SimpleResponse, data=input)
        else:
            await context.emit_event(ChatbotEvents.ComplexResponse, data=input)

# Step to generate a simple response using domain knowledge
class SimpleResponseStep(KernelProcessStep):
    GenerateSimpleResponse: ClassVar[str] = "generate_simple_response"
    
    @kernel_function(name=GenerateSimpleResponse)
    async def generate_simple_response(self, input, context: KernelProcessStepContext):
        prompt = """
        You are an expert in UAV domain. Provide concise response to the user query. 
        If the query is not related to UAV unless it is a greeting, then respond with "I am not sure about that. Please ask me about UAV related queries."
        """
        # Call GPT to get UAV-specific response
        response = client.chat.completions.create(
            model="gpt-4o",
            temperature=0,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps(input["query"])},
            ],
        ).choices[0].message.content

        return response

# Step to generate a plan for complex UAV log file analysis
class PlannerStep(KernelProcessStep):
    CreatePlan: ClassVar[str] = "create_plan"
    
    @kernel_function(name=CreatePlan)
    async def create_plan(self, input, context: KernelProcessStepContext):
        log_summary = input["file_summary"]
        prompt = f""" You are given the .bin file content from UAV flight log. Here is all the keys present in the file. Based on this keys, you have to create a plan on how data can be pulled based on user query.
        Provide a bullet point for the solid plan with minimum efforts.
        {log_summary}

        for the response follow this pattern. For the fields, only keep the abbreviations in the list:
        Fields
        - BARO
        - GPS
        Plans
        - Step 1
        - Step 2
        """
        # Call GPT to generate a plan based on log summary
        response = client.chat.completions.create(
            model="gpt-4o",
            temperature=0,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps(input["query"])},
            ],
        )
        plan = response.choices[0].message.content

        # Check if plan includes the required structure
        if "fields" not in plan.lower():
            await context.emit_event(ChatbotEvents.SimpleResponse, data=input)
        else:
            await context.emit_event(ChatbotEvents.PlanResponse, data={**input, "plan": plan})

class ExtractFieldStep(KernelProcessStep):
    ExtractFields: ClassVar[str] = "extract_fields"
        
    @kernel_function(name=ExtractFields)
    async def extract_fields(self, input, context: KernelProcessStepContext):
        plan = input["plan"]

        fields_section = re.search(r"Fields\s*(.*?)\n\s*Plans", plan, flags=re.DOTALL)
        fields_text = fields_section.group(1)
        fields = re.findall(r"-\s*(\w+)", fields_text)
        return {"session_id": input["session_id"],"fields":fields}

class FindSectionStep(KernelProcessStep):
    FindSessionInfo: ClassVar[str] = "find_section_info"
        
    @kernel_function(name=FindSessionInfo)
    async def find_section_info(self, data, context: KernelProcessStepContext):
        # URL of the webpage
        fields = data["fields"]
        url = "https://ardupilot.org/plane/docs/logmessages.html"

        # Send a GET request to the webpage
        response = requests.get(url)
        response.encoding = 'utf-8'
        response.raise_for_status()

        # Parse the HTML content
        soup = BeautifulSoup(response.text, 'html.parser')
        # Store found sections in a list
        sections = []

        for section_id in fields:
            section = soup.find(id=section_id.lower())
            if section:
                sections.append(section)
        return sections

class SampleExtractionStep(KernelProcessStep):
    SampleExtraction: ClassVar[str] = "sample_extraction"
        
    @kernel_function(name=SampleExtraction)
    async def sample_extraction(self, data, context: KernelProcessStepContext):
        session_id = data["session_id"]
        fields = data["fields"]
        log_file_path = f'uploads/{session_id}.bin'

        # Open the log
        mav = mavutil.mavlink_connection(log_file_path)

        # Dictionary to store the first occurrence of each field
        first_entries = {}

        # Loop until we get one of each or reach end of file
        while len(first_entries) < len(fields):
            msg = mav.recv_match(type=fields, blocking=False)
            if msg is None:
                continue
            msg_type = msg.get_type()
            if msg_type not in first_entries:
                first_entries[msg_type] = msg.to_dict()
        return first_entries

class CreateFunctionStep(KernelProcessStep):
    CreateFunction: ClassVar[str] = "create_function"
        
    @kernel_function(name=CreateFunction)
    async def create_function(self, plan, sections, samples, context: KernelProcessStepContext):
        
        prompt=""" Response only with python function where is input parameter is MAVLink connection object extracted from "from pymavlink import mavutil". and output the response that user want.
        use mavlink_connection.recv_match to optimize the search to find messages.
        do not use timeout for mav connection.
        make sure to break the loop when there is no more message to process.
        use .get_type() to find the message type.
        convert the message to dict if necessary.
        return it as dict.

        Do not make any assumptions as this code will later run autonomous.
        """
        user_message = f"""

        user query: {plan["query"]}
        {plan["plan"]}

        Field Info: 
        {sections}

        sample of fields:
        {samples}
        """
        response = client.chat.completions.create(
            model="gpt-4o",
            temperature= 0,
            messages=[
                {"role": "system", 
                "content": prompt},
                {
                    "role": "user",
                    "content": user_message,
                },
            ],
        )
        raw_function = response.choices[0].message.content
        return {**plan, "samples": samples, "sections":sections, "raw_function": raw_function}

class RecreateFunctionStep(KernelProcessStep):
    RecreateFunction: ClassVar[str] = "recreate_function"
        
    @kernel_function(name=RecreateFunction)
    async def recreate_function(self, data, context: KernelProcessStepContext):
        
        prompt=""" Response only with python function where is input parameter is MAVLink connection object extracted from "from pymavlink import mavutil". and output the response that user want.
        use mavlink_connection.recv_match to optimize the search to find messages.
        do not use timeout for mav connection.
        make sure to break the loop when there is no more message to process.
        use .get_type() to find the message type.
        convert the message to dict if necessary.
        return it as dict.

        Do not make any assumptions as this code will later run autonomous.
        """
        user_message = f"""

        user query: {data["query"]}
        {data["plan"]}

        Fields Info: Refer here to see what each of the keys in fields means. This will help you construct the function.
        {data["sections"]}

        sample of fields:
        {data["samples"]}
        
        There was error while executing the previously generated function. Below is the function and the error message:
        Function:
        {data["raw_function"]}
        Error:
        {data["execution_error"]}
        """
        response = client.chat.completions.create(
            model="gpt-4o",
            temperature= 0,
            messages=[
                {"role": "system", 
                "content": prompt},
                {
                    "role": "user",
                    "content": user_message,
                },
            ],
        )
        raw_function = response.choices[0].message.content
        data["raw_function"] = raw_function
        return data

class ExecuteFunctionStepState(BaseModel):
    run_index: int|None = 0
    
class ExecuteFunctionStep(KernelProcessStep[ExecuteFunctionStepState]):
    ExecuteFunction: ClassVar[str] = "execute_function"
    state: ExecuteFunctionStepState = Field(default_factory=ExecuteFunctionStepState)
    
    async def activate(self, state: KernelProcessStepState[ExecuteFunctionStepState]):
        self.state = state.state
        
    @kernel_function(name=ExecuteFunction)
    async def execute_function(self, data, context: KernelProcessStepContext):
        if self.state.run_index < 3:
            self.state.run_index += 1
            try:
                func = data["raw_function"]
                session_id = data["session_id"]
                log_file_path = f'uploads/{session_id}.bin'
                
                match = re.search(r"```python\n(.*?)```", func, re.DOTALL)
                code_str = match.group(1)

                # Execute the code in a local namespace
                local_namespace = {}
                exec(code_str, globals(), local_namespace)

                # Automatically detect the first function defined
                function_object = None
                for obj in local_namespace.values():
                    if isinstance(obj, types.FunctionType):
                        function_object = obj
                        break
                mav = mavutil.mavlink_connection(log_file_path)
                result = function_object(mav)  # replace `input` with your actual input object
                await context.emit_event(process_event=ChatbotEvents.ExecutionSuccess, data= {**data, "query_result": result})
            except Exception as e:
                await context.emit_event(process_event=ChatbotEvents.ExecutionFail, data= {**data, "execution_error": e})
        else:
            result = "Could not produce a result after multiple retries."
            await context.emit_event(process_event=ChatbotEvents.ExecutionSuccess, data= {**data, "query_result": result})

class AnalyticalResponseStep(KernelProcessStep):
    GenerateAnalyticalResponse: ClassVar[str] = "generate_analytical_response"
        
    @kernel_function(name=GenerateAnalyticalResponse)
    async def generate_analytical_response(self, input, context: KernelProcessStepContext):
        prompt = """
        You will be given a user query and the result. Prepare the concise response to pass back to chatbot.
        Provide the evidence or numerical data if present in the result. 
        Include the fields that was used in the plan.
        """

        user_input = f"""
        Here is user message:
        {input["query"]}

        here was the plan it executed:
        {input["plan"]}

        here is the function used:
        {input["raw_function"]}
        
        Here is the result:
        {input["query_result"]}
        """
        response = client.chat.completions.create(
            model="gpt-4o",
            temperature=0,
            messages=[
                {"role": "system", 
                "content": prompt},
                {
                    "role": "user",
                    "content": user_input,
                },
            ],
        )
        final_response = response.choices[0].message.content
        return final_response

class FinalResponseStepState(BaseModel):
    final_response: str|None = None
    
class FinalResponseStep(KernelProcessStep[FinalResponseStepState]):
    OUTPUT_EVAL: ClassVar[str] = "output_eval"
    state: FinalResponseStepState = Field(default_factory=FinalResponseStepState)
    
    async def activate(self, state: KernelProcessStepState[FinalResponseStepState]):
        self.state = state.state
        
    @kernel_function(name=OUTPUT_EVAL)
    async def output_eval(self, response, context: KernelProcessStepContext):
        self.state.final_response = response
          
def build_process() -> KernelProcess:
    builder = ProcessBuilder(name="EvaluateNarration")
    
    #steps
    decider = builder.add_step(DeciderStep)
    simple_response = builder.add_step(SimpleResponseStep)
    planner = builder.add_step(PlannerStep)
    extract_field = builder.add_step(ExtractFieldStep)
    find_section = builder.add_step(FindSectionStep)
    sample_extraction = builder.add_step(SampleExtractionStep)
    create_function = builder.add_step(CreateFunctionStep)
    execute_function = builder.add_step(ExecuteFunctionStep)
    recreate_function = builder.add_step(RecreateFunctionStep)
    analytical_response = builder.add_step(AnalyticalResponseStep)
    final_response = builder.add_step(FinalResponseStep)
    

    builder.on_input_event(ChatbotEvents.StartProcess).send_event_to(target=decider)
    decider.on_event(ChatbotEvents.SimpleResponse).send_event_to(target=simple_response)
    simple_response.on_function_result(SimpleResponseStep.GenerateSimpleResponse).send_event_to(target=final_response).stop_process()
    decider.on_event(ChatbotEvents.ComplexResponse).send_event_to(target=planner)
    planner.on_event(ChatbotEvents.SimpleResponse).send_event_to(target=simple_response)
    planner.on_event(ChatbotEvents.PlanResponse).send_event_to(target= extract_field).send_event_to(target=create_function, parameter_name="plan")
    extract_field.on_function_result(ExtractFieldStep.ExtractFields).send_event_to(target=find_section).send_event_to(target=sample_extraction)
    find_section.on_function_result(FindSectionStep.FindSessionInfo).send_event_to(target=create_function, parameter_name="sections")
    sample_extraction.on_function_result(SampleExtractionStep.SampleExtraction).send_event_to(target=create_function, parameter_name="samples")
    create_function.on_function_result(CreateFunctionStep.CreateFunction).send_event_to(target=execute_function)
    execute_function.on_event(ChatbotEvents.ExecutionSuccess).send_event_to(target=analytical_response)
    execute_function.on_event(ChatbotEvents.ExecutionFail).send_event_to(target=recreate_function)
    recreate_function.on_function_result(RecreateFunctionStep.RecreateFunction).send_event_to(target=execute_function)
    analytical_response.on_function_result(AnalyticalResponseStep.GenerateAnalyticalResponse).send_event_to(target=final_response).stop_process()
    
    return builder.build()

async def run_process(session_id, message_history, file_summary):
    input = {
        "query": message_history,
        "session_id": session_id,
        "file_summary": file_summary,
    }
    
    process = build_process()
    async with await start_local_process(
        process=process,
        kernel=Kernel(),
        initial_event=KernelProcessEvent(
            id=ChatbotEvents.StartProcess,
            data=input,
            visibility=KernelProcessEventVisibility.Public,
        ),
    ) as process_context:
        # Retrieve final state
        process_state = await process_context.get_state()
        output_step_state: KernelProcessStepState[FinalResponseStepState] = next(
            (s.state for s in process_state.steps if s.state.name == "FinalResponseStep"), None
        )
        
        if output_step_state:
            # Final user-facing answer:
            final_response = output_step_state.state.final_response
            return final_response
        else:
            return "No response generated."
