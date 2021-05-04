from tqdm import tqdm_notebook

#Делает прогресс бар при выполнении цикла. Удобно для больших циклов
cnt = 0
temp = 0
for i, batch in enumerate(tqdm_notebook(trainloader)):
        # так получаем текущий батч
        X_batch, y_batch = batch
        cnt = i
        if i % 2000 == 1999:
          temp += 1 
