import io

from minio import Minio
from minio.error import ResponseError

def minio_connect(server_minio_host, server_minio_port, server_minio_region, server_minio_access_key, server_minio_secret_key):
    minioClient = Minio(server_minio_host + ':' + server_minio_port, region = server_minio_region,
                  access_key=server_minio_access_key,
                  secret_key=server_minio_secret_key,
                  secure=False)
    return minioClient

def minio_upload_file(minioClient, backet_name, file_name, data):
    check_backet = minioClient.bucket_exists(backet_name)

    if check_backet == False:
        minioClient.make_bucket(backet_name)

    if check_backet:
        response = minioClient.put_object(backet_name, file_name, io.BytesIO(data), length=len(data), part_size=10*1024*1024)
        # response.close()
        # response.release_conn()

    # return data

def minio_download_file(minioClient, backet_name, file_name, error_if_not_exist = True):
    check_backet = minioClient.bucket_exists(backet_name)
    data = ""

    if check_backet:
        
        if error_if_not_exist == False:
            try:
                response = minioClient.get_object(backet_name, file_name)
                data = response.data
                response.close()
                response.release_conn()
                
            except:
                pass
            
        else:
            response = minioClient.get_object(backet_name, file_name)
            data = response.data
            response.close()
            response.release_conn()

    return data

def minio_delete_file(minioClient, backet_name, file_name):
    minioClient.remove_object(backet_name, file_name)

# def minio_2_download_file(server_minio_host, server_minio_port, server_minio_access_key, server_minio_secret_key, backet_name, file_name):
#     # global server_minio_host
#     # global server_minio_port
#     # global server_minio_access_key
#     # global server_minio_secret_key
#     # global server_minio_backet_name

#     minioClient = Minio(server_minio_host + ':' + server_minio_port,
#                   access_key=server_minio_access_key,
#                   secret_key=server_minio_secret_key,
#                   secure=False)

#     # backet_name = server_minio_backet_name

#     # try:
#     check_backet = minioClient.bucket_exists(backet_name)
#     # except ResponseError as err:
#     #     print(err)

#     data = ""

#     if check_backet:
#         response = minioClient.get_object(backet_name, file_name)
#         data = response.data
#         response.close()
#         response.release_conn()
#         # Get data of an object.
#         # try:
#         #     response = minioClient.get_object(backet_name, imgname)
#         #     # Read data from response.
#         #     data = response.data
#         # except ResponseError as err:
#         #     print(err)
#         # finally:
#         #     response.close()
#         #     response.release_conn()

#     return data