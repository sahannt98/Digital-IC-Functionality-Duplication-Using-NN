from Logic_Function.Combinational_Logic.Fulladder import Fulladder
from Logic_Function.Combinational_Logic.AndGate import AndGate
from Logic_Function.Combinational_Logic.comman import comman

from MysqlConfigure import MysqlConfigure 
from Combinational import Combinational

if __name__ == '__main__':
    #........................................Initialization parameter..................................................................
    NumberOfElement = 1000
    input_len = 10
    output_len = 6
    

    comman_logic = comman(["A and B", "A or (B and C)", "A ^ D"],5000)
    X,Y,input_len,output_len = comman_logic.make_truth_table() #Make a dataset
    
    
    Mysql_Configure = MysqlConfigure(input_len,output_len)
    Mysql_Configure.PushData(X, Y, "New_Function") #Push the data for MYSQL
    X,Y = Mysql_Configure.GetData("New_Function") #Get the data from MYSQL


    Combinational(X, Y) #Train the Model