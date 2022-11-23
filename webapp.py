from flask import Flask, render_template, request
from Logic_Function.Combinational_Logic.CommonCombinational import CommonCombinational
from Logic_Function.Sequential_Logic.CommonSequential import CommonSequential
from MysqlConfigure import MysqlConfigure 
from CommonNN import CommonNN
import os

app = Flask(__name__)

@app.route("/")
@app.route("/home", methods=['GET', 'POST'])
def home():
    return render_template('Home.html')

@app.route("/make_truth_table", methods=['GET', 'POST'])
def make_truth_table():
    if request.method == 'POST':
        Length_of_dataset = int(request.form.get("Length_of_dataset"))
        Logic_funtion = request.form.get("Logic_funtion")
        Table_Name = request.form.get("Table_Name")

        try:
            comman_logic = CommonCombinational([i for i in Logic_funtion.split(",")],Length_of_dataset)
            X,Y,input_len,output_len = comman_logic.make_truth_table()
            try:
                x_len = int(request.form.get("input_len"))
                input_len = max(x_len, input_len)
            except:
                pass
            try:
                y_len = int(request.form.get("output_len"))
                output_len = max(y_len, output_len)  
            except:
                pass 
            Mysql_Configure = MysqlConfigure(input_len,output_len)
            Mysql_Configure.PushData(X, Y, Table_Name)
            return render_template('TableResult.html', massage="Created table", status = "correct_result")
        except Exception as e:
            return render_template('TableResult.html', massage=e, status = "error_result")

@app.route("/Training_model", methods=['GET', 'POST'])
def Training_model():
    if request.method == 'POST':
        Percentage = int(request.form.get("Percentage"))
        Number_of_epochs = int(request.form.get("Number_of_epochs"))
        Table_Name = request.form.get("Table_Name")
        #try:
        Mysql_Configure = MysqlConfigure()
        X,Y = Mysql_Configure.GetData(Table_Name)
        Model = CommonNN(X, Y, Percentage, Number_of_epochs)
        Accuracy = Model.getAccuracy()
        return render_template('result.html', massage=Accuracy, status = "correct_result")
        #except Exception as e:
            #return render_template('error.html', massage=e)
    else:
        return render_template('Train.html')

@app.route("/Upload_file", methods=['GET', 'POST'])
def Upload_file():
    if request.method == 'POST':
        file = request.files['file']
        file_name = file.filename
        number_of_input = int(request.form.get("Input_pin"))
        Table_Name = request.form.get("Table_Name")
        try:
            if not "Logic_Function\\Sequential_Logic\\temporary" in [x[0] for x in os.walk("Logic_Function")]:
                os.mkdir("Logic_Function\\Sequential_Logic\\temporary")
            file_path = os.path.join(r'.\\Logic_Function\\Sequential_Logic\\temporary', "result.txt")
            if file_name == "":
                massage = "Please choose the file"
                return render_template('uploadtextresult.html', massage=massage, status = "error_result")
            elif file_name.split(".")[-1].upper() not in ["TXT"]:
                massage = "Please choose the valid text file"
                return render_template('uploadtextresult.html', massage=massage, status = "error_result")
            else:
                file.save(file_path)
                Sequential_logic = CommonSequential('Logic_Function\\Sequential_Logic\\temporary\\result.txt',number_of_input)
                X,Y,input_len,output_len = Sequential_logic.make_truth_table()
                try:
                    x_len = int(request.form.get("input_len"))
                    input_len = max(x_len, input_len)
                except:
                    pass
                try:
                    y_len = int(request.form.get("output_len"))
                    output_len = max(y_len, output_len)  
                except:
                    pass 
                Mysql_Configure = MysqlConfigure(input_len,output_len)
                Mysql_Configure.PushData(X, Y, Table_Name)
                return render_template('uploadtextresult.html', massage="Created table", status = "correct_result")
        except Exception as e:
            return render_template('uploadtextresult.html', massage=e, status = "error_result")

    else:
        return render_template('uploadtext.html')

if __name__ == "__main__":
    os.system("start \"\" http://127.0.0.1:5000")
    app.run(port=5000)