
using DataFrames
using CSV
using Statistics
using Indicators
using StatsBase


# Load historical price data for the two assets
# data = CSV.File("historical_data.csv") |> DataFrame


# # Define the pairs trading algorithm
# function pairs_trading_algorithm(data::DataFrame, asset1::Symbol, asset2::Symbol, lookback_period::Int, deviation_threshold::Float64)
#     # Calculate the historical price ratio
#     historical_ratio = calculate_price_ratio(data, asset1, asset2, lookback_period)
    
#     # Calculate the current price ratio
#     current_ratio = data[end, asset1] / data[end, asset2]
    
#     # Check for deviation from historical average
#     if current_ratio > (1 + deviation_threshold) * historical_ratio
#         # Asset 1 is overvalued, consider a short position on asset 1 and a long position on asset 2
#         println("Generate Signal: Short $asset1, Long $asset2")
#     elseif current_ratio < (1 - deviation_threshold) * historical_ratio
#         # Asset 2 is overvalued, consider a long position on asset 1 and a short position on asset 2
#         println("Generate Signal: Long $asset1, Short $asset2")
#     else
#         # No significant deviation, no trade signal
#         println("No Trade Signal")
#     end
# end

function pairs_trading_algorithm(data, asset1, asset2, lookback_period; num_sd=2, zscore_threshold=1.96)
    # Calculate correlation (optional for pair selection)
    correlation = calculate_correlation(data, asset1, asset2)

    # Generate signal using preferred method
    signal_sd_bands = generate_signal_with_sd_bands(data, asset1, asset2, lookback_period, num_sd)  # or use generate_signal_with_zscore()
    signal_zscore = generate_signal_with_zscore(data, asset1, asset2, lookback_period)

    # ### Execute based on signal
    # if signal != "No Trade Signal"
    #     # Place orders (example implementation)
    #     entry_price = data[end, asset1] / data[end, asset2]  # Assuming simultaneous execution
    #     place_orders(asset1, asset2, entry_price, 0.05, 0.10)  # Example stop-loss and take-profit ratios
    # end
    println("Signal with Standard Deviation bands $signal_sd_bands")
    println("Signal with z score bands $signal_sd_bands")

    return 1
end


# Example usage
data = CSV.File("_dat/_full_data.csv") |> DataFrame
asset1 = :UAL
asset2 = :DAL
signal = pairs_trading_algorithm(data, asset1, asset2, 30)



# pairs_trading_algorithm(data, :Asset1, :Asset2, 30, 0.05)
