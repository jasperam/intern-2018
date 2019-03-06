import dataset_g as ds
import pandas as pd

#生成数据集，样本800，noise为0.2
circle = ds.classify_circle_data(800, 0.2)
two_g = ds.classify_two_gauss_data(800, 0.2)
spiral = ds.classify_spiral_data(800, 0.2)
xor = ds.classify_xor_data(800, 0.2)

#输出dataframe
df1 = pd.DataFrame(circle)
df1.to_csv('circle_1.csv')
df2 = pd.DataFrame(two_g)
df2.to_csv('twog_1.csv')
df3 = pd.DataFrame(spiral)
df3.to_csv('spiral_1.csv')
df4 = pd.DataFrame(xor)
df4.to_csv('xor_1.csv')