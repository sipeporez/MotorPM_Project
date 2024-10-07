from flask import Flask, render_template, request, jsonify
import websocket
import threading

app = Flask(__name__)
ws = None  # 웹소켓 객체를 전역 변수로 선언

# WebSocket 클라이언트 연결 설정
def on_message(ws, message):
    print(f"Received: {message}")

def on_error(ws, error):
    print(f"Error: {error}")

def on_close(ws, close_status_code, close_msg):
    print("Connection closed")

def on_open(ws):
    print("Connection opened")
    ws.send("Hello, Server!")

def start_websocket():
    global ws
    ws = websocket.WebSocketApp("ws://localhost:8080/ws",  # 스프링 부트 서버의 웹소켓 URL
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close)
    ws.on_open = on_open
    ws.run_forever()

# 웹소켓 클라이언트를 별도의 스레드로 실행
threading.Thread(target=start_websocket).start()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/send_message', methods=['POST'])
def send_message():
    global ws
    message = request.form['message']
    if ws:
        ws.send(message)  # 서버로 메시지 전송
        return jsonify({'status': 'success', 'message': f"Message sent: {message}"})
    else:
        return jsonify({'status': 'error', 'message': "WebSocket is not connected."})

if __name__ == "__main__":
    app.run(port=5000)
