from flask import Flask, render_template, request
from Logic_Function.Combinational_Logic.comman import comman
from MysqlConfigure import MysqlConfigure 
from Combinational import Combinational
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
            comman_logic = comman([i for i in Logic_funtion.split(",")],Length_of_dataset)
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
            return render_template('TableResult.html', massage="Created table")
        except:
            return render_template('TableResult.html', massage="Error come")

@app.route("/train_model", methods=['GET', 'POST'])
def Traing_model():
    if request.method == 'POST':
        Percentage = int(request.form.get("Percentage"))
        Number_of_epochs = int(request.form.get("Number_of_epochs"))
        Table_Name = request.form.get("Table_Name")
        try:
            Mysql_Configure = MysqlConfigure()
            X,Y = Mysql_Configure.GetData(Table_Name)
            Model = Combinational(X, Y, Percentage, Number_of_epochs)
            Acurassy = Model.getAcurasy()
            return render_template('result.html', massage=Acurassy)
        except:
            return render_template('error.html')
    else:
        return render_template('Train.html')

if __name__ == "__main__":
    os.system("start \"\" http://127.0.0.1:5000")
    app.run(port=5000)