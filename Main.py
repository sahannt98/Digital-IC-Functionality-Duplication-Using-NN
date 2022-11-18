from Fulladder import Fulladder
from AndGate import AndGate 
from MysqlConfigure import MysqlConfigure 
from Combinational import Combinational

if __name__ == '__main__':
    #Initialization parameter
    NumberOfElement = 100000
    input_len = 10
    output_len = 6
    
    #Make the object
    #Full_adder = Fulladder(NumberOfElement)
    Mysql_Configure = MysqlConfigure(input_len,output_len)

    #X,Y =Full_adder.Fulladder_gate() #Make a dataset
    #Mysql_Configure.PushData(X, Y, "Full_adder") #Push the data for MYSQL
    X,Y = Mysql_Configure.GetData("Full_adder") #Get the data from MYSQL

    Combinational(X, Y) #Train the Model