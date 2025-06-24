from pymavlink import mavutil

def count_field_points(session_id, field:str):
    count = 0

    log_file_path = f'uploads/{session_id}.bin'
    mav = mavutil.mavlink_connection(log_file_path)
    
    while True:
        message = mav.recv_match(type=field, blocking=False)
        if message is None:
            break
        count += 1
    return {'count': count}

def get_max_battery_temperature(session_id):
    max_temp = None

    log_file_path = f'uploads/{session_id}.bin'
    mav = mavutil.mavlink_connection(log_file_path)

    while True:
        message = mav.recv_match(type='BAT', blocking=False)
        if message is None:
            break

        message_dict = message.to_dict()
        current_temp = message_dict.get('Temp', None)

        if current_temp is not None:
            if max_temp is None or current_temp > max_temp:
                max_temp = current_temp

    return {'max_battery_temperature': max_temp}