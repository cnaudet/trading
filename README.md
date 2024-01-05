# trading
For JULIA CODE:
 must have the trading repository set to the julia load path, in bash this looks like 
 export JULIA_LOAD_PATH=$JULIA_LOAD_PATH:/path/to/trading/_julia_files
 can create and put executable into a .bashrc file in home directory and source it to call it to path
 source ~/.bashrc

 then, opening julia in the trading/_julia_files directory, one can run 
 using Revise, TradeMe.jl

 Assuming you have all packages installed, it should run smoothly and you will be able to run all functions included



 FOR PYTHON CODE:
 It is written in python, using pyomo, and is designed for us in google colab. It also assumes a folder called 
 'Colab Notebooks/data/_stock_history/' which is embedded under the Colab Notebooks default folder. You can import your own data here and use the code, assuming it has the same naming conventions as those specified in the code. The data used and convention used comes from Yahoo Finance stock history data. 