# module MachineLearningTrading

# using Pkg
# Pkg.add("MLJ")
# using CSV, DataFrames, Dates

export load_historical_data, preprocess_data, create_features, split_data

export generate_signals, generate_artificial_signals
export build_model, train_model, evaluate_model


function load_historical_data(symbol)
    # url = "https://example.com/historical_data/$symbol.csv"  # Replace with actual data source
    # data = CSV.read(url, DataFrame)
    data = CSV.File("_dat/$symbol.csv") |> DataFrame
    # data.Date = Dates.Date.(data.Date)  # Convert date column to Date type
    return select(data, [:Date, :Open, :High, :Low, :Close, :Volume])
end


function preprocess_data(data)
    data = dropmissing(data)  # Remove rows with missing values
    data[!,:standardizedClose] = standardize(data.Close)  # Standardize the Close price using zscore
    return data
end

function clean_feature_data(data0; ma2=20)
    return data0[ma2+1:end, :]  # Remove rows with zero values from computing EMA
end

function standardize(x)
    μ = mean(x)
    σ = std(x)
    return (x .- μ) ./ σ
end


function create_features(data; ma1=5, ma2=20)
    data[!, :MA5] = [zeros(ma1-1); MarketTechnicals.sma(data[!,:standardizedClose], ma1)[:]]  # Calculate 5-day moving average
    data[!, :MA20] = [zeros(ma2-1); MarketTechnicals.sma(data[!,:standardizedClose], ma2)[:]]  # Calculate 20-day moving average

    data[!, :momentum14] = momentum(data[:, :standardizedClose]; n=14)


    time_array = TimeArray(data; timestamp=:Date)
    t_array_close = time_array[:, :standardizedClose]
    m = 14
    rsi_m = TimeSeries.values(MarketTechnicals.rsi(t_array_close, m))[:]
    data[!, :RSI14] = [zeros(m); rsi_m]  # Calculate 14-day RSI

    data = clean_feature_data(data; ma2)

    return data
end


function split_data(features, targets; split_ratio=0.8)
    return partition(features, targets, 0.8)
    # return features[train, :], targets[train], features[test, :], targets[test]
end

function partition(features, targets, split_ratio=0.8)
    # Get the number of samples
    n = size(features, 1)

    # Create a random permutation of indices
    indices = shuffle(1:n)

    # Calculate the number of samples for the training set
    n_train = round(Int, split_ratio * n)

    # Split the indices into training and test sets
    train_indices = indices[1:n_train]
    test_indices = indices[n_train+1:end]

    # Use the indices to extract the corresponding rows from features and targets
    train_features, train_targets = features[train_indices, :], targets[train_indices]
    test_features, test_targets = features[test_indices, :], targets[test_indices]

    return train_features, train_targets, test_features, test_targets
end



function build_model(::Chain; ninput=2, noutput=3)
    # Define your machine learning model here



    # Decision tree
    # model =  DecisionTreeClassifier()

    # Neural NetworkN
    # Define the model
    model = Chain(
        Dense(ninput, 64, tanh),   # Input layer with ninput features, output size 64, tanh activation
        Dense(64, 32, tanh),  # Hidden layer with output size 32, tanh activation
        Dense(32, noutput),         # Output layer with noutput categories (target classes), linear activation
        softmax)              # Softmax activation for converting raw outputs to probabilities

    return model
end

function train_model(::DecisionTreeClassifier, features, targets; plt_losses=false)
    # Train the model
    # machine = machine(model, features, categorical(targets))
    # machine_var = machine(model, features, targets)

    ### try this
    tree = (@load DecisionTreeClassifier pkg=DecisionTree verbosity=0)()
    forest = EnsembleModel(model=tree, n=100);
    machine_var = machine(forest, features, categorical(targets))

    trained_machine = MLJ.fit!(machine_var)
    convergence_info = report(trained_machine)
    println("Convergence information: $convergence_info")
    return trained_machine
end

export onehot_encode
# Convert the target labels to a one-hot encoding matrix
function onehot_encode(labels)
    unique_labels = unique(labels)
    num_classes = length(unique_labels)
    encoded_labels = Flux.onehotbatch(labels, unique_labels)
    return encoded_labels
end

function train_model(model0::Chain, features_dt, targets_dt; plt_losses=false)
    ns, nf = size(features_dt)

    features = Matrix{Float64}(undef, ns, nf)
    targets = targets_dt # Vector{Int64}(undef, ns)

    # extract vectors from data table
    i = 1
    for sym in names(features_dt)
        features[:, i] = features_dt[:, sym]
        i += 1
    end
    features = features'


    # Train the model
    # machine = machine(model, features, categorical(targets))
    # machine_var = machine(model, features, targets)

    ### try this
    # tree = (@load DecisionTreeClassifier pkg=DecisionTree verbosity=0)()
    # forest = EnsembleModel(model=tree, n=100);

    # Define our model, a multi-layer perceptron with one hidden layer of size 3:
    # using Flux

    # target = Flux.onehotbatch(targets, [1, 0, -1])   
    # target_var = [1, 0, -1]
    # target = Flux.onehotbatch(target, [1, 0, -1])

    encoded_targets = onehot_encode(targets)
    loader = Flux.DataLoader((features, encoded_targets), batchsize=64, shuffle=true);
    learning_rate= 0.01
    optim = Flux.setup(Flux.Adam(learning_rate), model0)  # will store optimiser momentum, etc. with 0.01 learning rate

    # Training loop, using the whole data set 1000 times:
    losses = []
    @showprogress for epoch in 1:1_000
        for (x, y) in loader
            loss, grads = Flux.withgradient(model0) do m
                # Evaluate model and loss inside gradient context:
                y_hat = m(x)
                Flux.crossentropy(y_hat, y)
                
            end
            Flux.update!(optim, model0, grads[1])
            push!(losses, loss)  # logging, outside gradient context
        end
    end

    println(plt_losses)
    if plt_losses
        plot_loss_data(losses, loader)
    end



    # machine_var = machine(model, features, categorical(targets))

    # trained_machine = MLJ.fit!(machine_var)
    # convergence_info = report(trained_machine)
    # println("Convergence information: $convergence_info")
    return model0, losses, loader
end

function evaluate_model(trained_machine, features, targets)
    # Evaluate the model
    ŷ = MLJ.predict(trained_machine, features)
    acc = MLJ.accuracy(ŷ, targets)
    return acc
end

function evaluate_model(trained_model::Chain, features_dt, targets_dt)
    ns, nf = size(features_dt)
    features = Matrix{Float64}(undef, ns, nf)
    targets = targets_dt # Vector{Int64}(undef, ns)

    # extract vectors from data table
    i = 1
    for sym in names(features_dt)
        features[:, i] = features_dt[:, sym]
        i += 1
    end
    features = features'


    # Evaluate the model
    out2 = trained_model(features) # first row is prob. of true, second row p(false)

    # test_target 
    # out2 = out2[:]
    println(size(features_dt))
    println(size(targets_dt))
    println("size of output",  (size(out2)))
    test_target = clean_categorical_signal_data(out2)
    acc = mean(test_target .== targets) 
    # println(out2)

    return acc
end

export clean_categorical_signal_data
function clean_categorical_signal_data(target_cat)
    println(size(target_cat))
    nf, ns = size(target_cat)
    test_target = Vector{Float64}(undef, ns)

    for s in 1:ns
        for i in 1:3
            println(target_cat[:, s])
            if target_cat[i, s]
                if i == 1
                    test_target[s] = 0
                 elseif i == 2 
                    test_target[s] = -1
                 elseif i == 3
                    test_target[s] = 1
                end
            end
        end
    end
end

function generate_signals(trained_machine, features)
    # Generate trading signals
    signals = MLJ.predict(trained_machine, features)
    return signals
end


function generate_artificial_signals(data)
    # Example: Buy when 5-day MA crosses above 20-day MA, Sell when RSI < 30
    data[!, :Signal] .= 0  # Initialize signals column

    # Generate Buy signals
    data[(data[1:end, :MA5] .> data[1:end, :MA20]) .& (data[1:end, :RSI14] .> 30), :Signal] .= 1

    # Generate Sell signals
    data[(data[1:end, :RSI14] .< 30), :Signal] .= -1

    return data
end

function generate_artificial_signals_from_mom(data; buy_threshold=0.2, sell_threshold=-0.05)
    # Assuming you have a DataFrame with a :Date and :Momentum column
    # Create a new column for signals and initialize it with "Hold" (0)
    data[!, :Signal] = fill(0, nrow(data))

    # Generate Buy signals
    data[(data[1:end, :MA5] .> data[1:end, :MA20]) .& (data[1:end, :momentum14] .> buy_threshold), :Signal] .= 1

    # Generate signals based on Momentum values
    # data[!, :Signal][data[!, :Momentum] > buy_threshold] .= 1  # Buy signal
    data[(data[1:end, :MA5] .< data[1:end, :MA20]) .& (data[1:end, :momentum14] .< sell_threshold), :Signal] .= -1  # Sell signal

    return data
end

export generate_profit_loss_from_sig
function generate_profit_loss_from_sig(data; cash=10000.0, close_col=:Close)
    # Initialize variables
    stocks_held = 0
    capital = 0.0
    pnl = []

    # Iterate through the data
    for row in 1:size(data, 1)
        signal = data[row, :Signal]

        # Buy signal
        println("Day $row Signal $signal: Cash held: $cash, stocks held $stocks_held")
        if signal == 1
            if cash > 0
                price = data[row, close_col]
                stocks_to_buy = cash / price
                stocks_held += stocks_to_buy
                cash -= stocks_to_buy * data[row, close_col]
                println("Buying $stocks_to_buy stocks at $price each")
            end
        end

        # Sell signal
        if signal == -1 && stocks_held > 0
            price = data[row, close_col]
            cash += stocks_held * price
            println("Selling $stocks_held stocks for $price each")
            stocks_held = 0
        end

        # Calculate capital and store it
        capital = cash + stocks_held * data[row, close_col]
        push!(pnl, capital)
    end

    data[!, :profit_loss] = pnl

    return data, pnl
end








# ... other functions (model selection, training, evaluation, signal generation)

# end
