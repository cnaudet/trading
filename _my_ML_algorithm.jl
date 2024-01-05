export run_ML

function run_ML(; symbol = "SPY")
# Load Historical Data
historical_data = load_historical_data(symbol)

# Preprocess Data
preprocessed_data = preprocess_data(historical_data)

# Create Features
feature_enriched_data_nosig = create_features(preprocessed_data)

# Generate signals based on RSI and EMA
# feature_enriched_data = generate_artificial_signals(feature_enriched_data)
feature_enriched_data = generate_artificial_signals_from_mom(feature_enriched_data_nosig; sell_threshold=-0.15, buy_threshold=0.1)


# Split Data
features, targets = feature_enriched_data[:, [:MA5, :momentum14, :Close]], feature_enriched_data[:, :Signal]  # Adjust columns accordingly
train_features, train_targets, test_features, test_targets = split_data(features, targets)

# Build Model
nf = size(features,2)
ninput=nf
model = build_model(Chain(); ninput)

# Step 6: Train Model
plt_losses=true
trained_model, losses, loader = train_model(model, train_features, train_targets; plt_losses=false)
# plot_loss_data(losses, loader)
# plt_all_relevant_features(feature_enriched_data)

# Step 7: Evaluate Model
# my_accuracy = evaluate_model(trained_model, test_features, test_targets)
# println("Model Accuracy: $my_accuracy")

# Step 8: Generate Signals
# signals = generate_signals(trained_model, test_features)

# ... Additional steps for backtesting, strategy execution, etc.

return feature_enriched_data, trained_model, losses, loader

end
