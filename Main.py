from Fulladder import Fulladder
from AndGate import AndGate 
from MysqlConfigure import MysqlConfigure 
from Combinational import Combinational

if __name__ == '__main__':
    #Inizilization perameter
    NumberOfElement = 1000
    input_len = 50
    output_len = 10
    
    #Make the object
    And_Gate = AndGate(NumberOfElement)
    Mysql_Configure = MysqlConfigure(input_len,output_len)

    X,Y =And_Gate.And_Get() #Make a dataset
    Mysql_Configure.PushData(X, Y, "andgate") #Push the data for MYSQL
    X,Y = Mysql_Configure.GetData("andgate") #Get the data from MYSQL

    Combinational(X, Y) #Train the Model
