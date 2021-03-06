import seaborn as sns

def seaborn_visible_dataframe(df, **args):
    '''
    Рисует таблицу графиков
    data, hue=None, hue_order=None, palette=None, vars=None, x_vars=None, y_vars=None,
    kind="scatter", diag_kind="auto", markers=None, height=2.5, aspect=1, corner=False,
    dropna=False, plot_kws=None, diag_kws=None, grid_kws=None, size=None
    '''
    sns.set(font_scale=1.3) #Немного увеличивает шрифт
    sns.pairplot(df, **args)
    

def seaborn_visible_heatmap(data):
    # Построим heatmap для оценки корреляции численных данных
    sns.heatmap(data.corr(), annot=True, cmap= 'coolwarm', linewidths=3, linecolor='black')
    
def seaborn_visible_pairplot(data, target_label_name_col):
    # Построим pairplot для оценки отношения между всеми парами численных признаков 
    sns.pairplot(data, hue=target_label_name_col)
