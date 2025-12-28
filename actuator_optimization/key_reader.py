import sys
import ast

def model_info(model_num):

    with open('C:/Users/and008/Documents/Models V2/' + "key_file.txt", 'r') as f_r:
        lines = f_r.readlines()
        for line in lines:
            curr_dict = ast.literal_eval(line)
            if int(curr_dict['key']) == model_num:
                print("MODEL " + str(model_num))
                for key, value in curr_dict['actuator']['geometry'].items():
                    print(key + ': ' + str(value))

if __name__ == "__main__":
    model_num = int(sys.argv[1])
    model_info(model_num)