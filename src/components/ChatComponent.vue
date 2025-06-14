<template>
    <div class="chat-container">
        <div v-if="!isMinimized" class="chat-window">
            <div class="chat-header">
                <span class="title">Flight Assistant</span>
                <div class="header-buttons">
                    <button @click="clearChat" class="clear-btn" title="Clear chat">🧹</button>
                    <button @click="isMinimized = true" class="minimize-btn" title="Minimize">−</button>
                </div>
            </div>

            <div class="chat-body">
                <div class="input-area">
                    <div class="preset-messages" v-if="!hasUserSentMessage">
                        <button
                            v-for="preset in presetMessages"
                            :key="preset"
                            @click="selectPreset(preset)"
                        >
                            {{ preset }}
                        </button>
                    </div>

                    <textarea
                        v-model="newMessage"
                        ref="messageInput"
                        :class="['message-input', { scrollable: isAtMaxHeight }]"
                        @input="autoResize"
                        @keydown.enter.exact.prevent="sendMessage"
                        placeholder="Ask a question about this flight..."
                    />
                </div>

                <div class="chat-messages" ref="chatMessages">
                    <transition-group name="chat" tag="div">
                        <div
                            v-for="msg in messages"
                            :key="msg.id"
                            :class="['message-wrapper', msg.sender]"
                        >
                            <div class="message-bubble">
                                <div class="text">{{ msg.text }}</div>
                                <div class="timestamp">{{ formatTime(msg.timestamp) }}</div>
                            </div>
                        </div>
                    </transition-group>

                    <div v-if="isBotTyping" class="message-wrapper bot">
                        <div class="message-bubble typing">Bot is typing...</div>
                    </div>
                </div>
            </div>
        </div>

        <div v-else class="chat-icon" @click="isMinimized = false">
            💬
        </div>
    </div>
</template>

<script>
import { v4 as uuidv4 } from 'uuid'

export default {
    data () {
        return {
            messages: [],
            newMessage: '',
            isBotTyping: false,
            isMinimized: true,
            hasUserSentMessage: false,
            isAtMaxHeight: false,
            socket: null,
            wsUrl: 'ws://localhost:8000/chat',
            presetMessages: [
                'What was the highest altitude reached during the flight?',
                'When did the GPS signal first get lost?',
                'When was the first instance of RC signal loss?',
                'Can you spot any issues in the GPS data?'
            ]
        }
    },
    mounted () {
        this.initWebSocket()
        this.initBotGreeting()
        this.scrollToBottom()
    },
    methods: {
        initWebSocket () {
            this.socket = new WebSocket(this.wsUrl)

            this.socket.onopen = () => console.log('WebSocket connected')
            this.socket.onmessage = (event) => {
                this.messages.push({
                    id: uuidv4(),
                    sender: 'bot',
                    text: event.data,
                    timestamp: new Date().toISOString()
                })
                this.isBotTyping = false
                this.$nextTick(this.scrollToBottom)
            }
            this.socket.onerror = (err) => console.error('WebSocket error:', err)
            this.socket.onclose = () => {
                console.warn('WebSocket closed. Reconnecting...')
                setTimeout(this.initWebSocket, 3000)
            }
        },
        formatTime (isoString) {
            const date = new Date(isoString)
            return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
        },
        scrollToBottom () {
            const container = this.$refs.chatMessages
            if (container) {
                container.scrollTop = container.scrollHeight
            }
        },
        initBotGreeting () {
            this.messages.push({
                id: uuidv4(),
                sender: 'bot',
                text: 'Hello! I’m here to help you investigate MAVLink flight data. Start typing your question below.',
                timestamp: new Date().toISOString()
            })
            this.$nextTick(this.scrollToBottom)
        },
        selectPreset (text) {
            this.newMessage = text
            this.sendMessage()
            this.$nextTick(this.scrollToBottom)
        },
        clearChat () {
            this.messages = []
            this.hasUserSentMessage = false
            this.initBotGreeting()
        },
        async sendMessage () {
            const trimmed = this.newMessage.trim()
            if (!trimmed || !this.socket || this.socket.readyState !== WebSocket.OPEN) return

            if (!this.hasUserSentMessage) {
                this.hasUserSentMessage = true
            }

            const userMsg = {
                id: uuidv4(),
                sender: 'user',
                text: trimmed,
                timestamp: new Date().toISOString()
            }
            this.messages.push(userMsg)
            this.newMessage = ''
            this.isAtMaxHeight = false
            this.$nextTick(() => {
                const textarea = this.$refs.messageInput
                if (textarea) {
                    textarea.style.height = 'auto'
                }
                this.scrollToBottom()
            })

            this.isBotTyping = true

            try {
                const payload = {
                    sessionId: window.sessionUniqueId,
                    message: trimmed
                }
                this.socket.send(JSON.stringify(payload))
            } catch (err) {
                this.messages.push({
                    id: uuidv4(),
                    sender: 'bot',
                    text: 'Error communicating with server.',
                    timestamp: new Date().toISOString()
                })
                this.$nextTick(this.scrollToBottom)
            } finally {
                this.isBotTyping = false
            }
        },
        autoResize () {
            const textarea = this.$refs.messageInput
            if (textarea) {
                textarea.style.height = 'auto'
                const newHeight = textarea.scrollHeight
                const maxHeight = 120
                textarea.style.height = Math.min(newHeight, maxHeight) + 'px'
                this.isAtMaxHeight = newHeight > maxHeight
            }
        }
    }
}
</script>

<style scoped>
.chat-container {
    position: fixed;
    bottom: 20px;
    right: 20px;
    font-family: 'Nunito Sans', sans-serif;
    z-index: 1000;
}

.chat-window {
    width: 360px;
    height: 500px;
    background: #ffffff;
    border: 1px solid #ddd;
    border-radius: 12px;
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.15);
    overflow: hidden;
    display: flex;
    flex-direction: column;
}

.chat-header {
    background: #f5f5f5;
    padding: 10px 12px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid #ddd;
    font-weight: bold;
}

.chat-header .title {
    font-size: 16px;
}

.header-buttons button {
    background: none;
    border: none;
    cursor: pointer;
    margin-left: 10px;
    font-size: 16px;
}

.chat-body {
    padding: 12px;
    flex: 1;
    display: flex;
    flex-direction: column-reverse;
    overflow: hidden;
}

.input-area {
    display: flex;
    flex-direction: column;
    gap: 8px;
    margin-top: 10px;
}

.chat-messages {
    max-height: 350px;
    overflow-y: auto;
    margin-bottom: 10px;
    scroll-behavior: smooth;
    padding-right: 5px;
    flex: 1;
}

.message-wrapper {
    display: flex;
    margin: 8px 0;
}

.message-wrapper.user {
    justify-content: flex-end;
}

.message-wrapper.bot {
    justify-content: flex-start;
}

.message-bubble {
    max-width: 70%;
    background: #e6f0ff;
    color: #007bff;
    padding: 10px;
    border-radius: 12px;
    word-wrap: break-word;
    animation: fadeInSlide 0.3s ease;
}

.message-wrapper.bot .message-bubble {
    background: #eafbe7;
    color: #28a745;
}

.message-input {
    width: 100%;
    min-height: 42px;
    max-height: 120px;
    padding: 10px;
    border: 1px solid #ccc;
    border-radius: 8px;
    font-size: 14px;
    resize: none;
    overflow-y: hidden;
    box-sizing: border-box;
    line-height: 1.4;
    transition: height 0.2s ease-in-out;
}

.message-input.scrollable {
    overflow-y: auto;
}

.typing {
    font-style: italic;
    color: #999;
    padding: 6px 10px;
}

.timestamp {
    font-size: 10px;
    color: #888;
    margin-top: 4px;
    text-align: right;
}

.preset-messages {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin: 10px 0;
}

.preset-messages button {
    flex: 1 1 auto;
    padding: 6px 12px;
    background: #f0f0f0;
    border: 1px solid #ccc;
    border-radius: 6px;
    cursor: pointer;
    font-size: 13px;
    transition: background 0.2s ease-in-out;
}

.preset-messages button:hover {
    background: #e0e0e0;
}

.chat-icon {
    width: 50px;
    height: 50px;
    background: #007bff;
    color: white;
    font-size: 24px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 50%;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
    cursor: pointer;
    transition: transform 0.2s;
}

.chat-icon:hover {
    transform: scale(1.05);
}

@keyframes fadeInSlide {
    0% {
        opacity: 0;
        transform: translateY(10px);
    }
    100% {
        opacity: 1;
        transform: translateY(0);
    }
}

.chat-enter-active,
.chat-leave-active {
    transition: all 0.3s ease;
}

.chat-enter-from,
.chat-leave-to {
    opacity: 0;
    transform: translateY(10px);
}
</style>
