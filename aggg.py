import socket
import threading

# İstemci ayarları
nickname = input("Choose your nickname: ")

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(('10.77.137.13', 12345))

# Sunucudan gelen mesajları dinleme
def receive():
    while True:
        try:
            message = client.recv(1024).decode('utf-8')
            if message == 'NICK':
                client.send(nickname.encode('utf-8'))
            else:
                print(message)
        except:
            # Bağlantı hatası olduğunda istemciyi kapat
            print("An error occurred!")
            client.close()
            break

# Mesaj gönderme fonksiyonu
def write():
    while True:
        message = f'{nickname}: {input("")}'
        client.send(message.encode('utf-8'))

# Dinleme ve yazma thread'lerini başlat
receive_thread = threading.Thread(target=receive)
receive_thread.start()

write_thread = threading.Thread(target=write)
write_thread.start()