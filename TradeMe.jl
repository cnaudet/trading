module TradeMe


using Reexport

@reexport using DataFrames
@reexport using CSV
@reexport using Statistics
@reexport using Indicators
@reexport using StatsBase
@reexport using TimeSeries
@reexport using MarketTechnicals
@reexport using DecisionTree
@reexport using Indicators
@reexport using Plots
@reexport using Random
@reexport using MLJ
@reexport using DecisionTree
@reexport using Flux
@reexport using XGBoost
@reexport using ProgressMeter
@reexport using Indicators
# Decision Tree loading
@load DecisionTreeClassifier pkg=DecisionTree
# Support Vector Machine Loading
# @load SVMClassifier
# Logistic Regression
# @load LogisticClassifier pkg=MLJScikitLearnInterface
# KNN Nearest neighbors
@load KNNClassifier


include("_auxiliary_functions.jl")
include("_machine_learning_trading.jl")
include("_my_ML_algorithm.jl")
include("_plt_tools.jl")
# include("_auxiliary_functions.jl")


end


