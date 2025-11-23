import requests, os


def send_csv_to_url(csv_file_path: str, url: str) -> bool:
    try:
        # Verificar que el archivo existe
        if not os.path.exists(csv_file_path):
            print(f"Error: El archivo {csv_file_path} no existe")
            return False
        
        # Preparar el archivo para enviar
        with open(csv_file_path, 'rb') as csv_file:
            files = {'file': (os.path.basename(csv_file_path), csv_file, 'text/csv')}
            
            # Hacer la petición POST
            response = requests.post(url, files=files)
            
            # Verificar si fue exitoso
            if response.status_code == 200:
                print(f"Archivo enviado exitosamente. Respuesta: {response.text}")
                return True
            else:
                print(f"Error en la petición. Código: {response.status_code}, Respuesta: {response.text}")
                return False
                
    except Exception as e:
        print(f"Error al enviar el archivo: {e}")
        return False


if(__name__=="__main__"):  
    file = "tests/prueba.csv"
    api = "http://127.0.0.1:8000/api/v1/"
    send_csv_to_url(file,f"{api}datasets")