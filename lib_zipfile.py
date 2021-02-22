import zipfile 

def arhive_brands_script_to_zip(submission_name):
    '''
    Процедура архивирует два файла в zip архив
    
    submission_name = 't2_sub/submission.zip'
    '''


    compression = zipfile.ZIP_DEFLATED

    with zipfile.ZipFile(submission_name, 'w') as zipObj:
        for filename in [
            'brands_re',
            'script.py',
        ]:
            zipObj.write(
                f'arhiv/{filename}', 
                arcname=filename, 
                compress_type=compression
            )
        print(zipObj.namelist())
        
        
    