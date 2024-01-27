import os
import random
class_name_list = os.listdir("第四届人工智能挑战赛/dataset")
random_class_names = []
for i in range(563):
    random_class_num = random.randint(0, 13)
    random_class_names.append(class_name_list[random_class_num])
print("随机选取的563个类别为：", random_class_names)