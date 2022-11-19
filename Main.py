from Logic_Function.Combinational_Logic.Fulladder import Fulladder
from Logic_Function.Combinational_Logic.AndGate import AndGate 
from MysqlConfigure import MysqlConfigure 
from Combinational import Combinational

if __name__ == '__main__':
    #........................................Initialization parameter..................................................................
    NumberOfElement = 1000
    input_len = 10
    output_len = 6
    
    #.........................................Make the object.........................................................................
    Full_adder = Fulladder(NumberOfElement,5)
    Mysql_Configure = MysqlConfigure(input_len,output_len)

    #.........................................Handle dataset..........................................................................
    X,Y =Full_adder.Fulladder_gate() #Make a dataset
    Mysql_Configure.PushData(X, Y, "Full_adder") #Push the data for MYSQL
    X,Y = Mysql_Configure.GetData("Full_adder") #Get the data from MYSQL

    #........................................Trainthe Model...........................................................................
    Combinational(X, Y, input_len, output_len) #Train the Model