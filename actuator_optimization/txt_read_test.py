
import ast

# with open("key_file.txt", 'rb') as f_r:
#     data = f_r.read()
#     print(data)
#     d = pickle.loads(data)
#     print(d)

with open("key_file.txt", 'r') as f_r:
    lines = f_r.readlines()
    for line in lines:
        dict = ast.literal_eval(line)
        print(dict)
        print(type(dict))

        # if even_odd_counter % 2 == 0:
        #     print(even_odd_counter)
        #     print(line)
        #     data = pickle.loads(line)
        #     print(data)
        # even_odd_counter +=1