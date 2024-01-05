export plt_all_relevant_features, plot_loss_data

function plt_all_relevant_features(features)
    # Assuming 'features' is your DataFrame and 'plt' is the main plot
    plt0 = plt_close_data(features)

    plt1 = plt_mv_day_averages(features)

    plt2 = plt_momentum_indicator(features)

    plt3 = plt_signal_data(features)

    features, pnl = generate_profit_loss_from_sig(features)

    plt4 = plt_pnl_data(features)
    # Layout the plots vertically
    plot(plt0, plt1, plt2, plt3, plt4, layout=(5, 1))
end

function plt_pnl_data(features)
    plt = plot(features[:, :profit_loss],  color=:black, markersize=1)

    plot!(plt, title="Capital Data", legend=false)
    return plt
end

function plt_close_data(features)
    plt = plot(features[:, :Close],  color=:black, markersize=1)

    plot!(plt, title="Close Data", legend=false)
    return plt
end

function plt_signal_data(features)
    plt = plot(features[:, :Signal], seriestype=:scatter, color=:red, markersize=2)

    plot!(plt, title="Artificial signal", legend=false)
    return plt
end
export plt_mv_day_averages
function plt_mv_day_averages(features)

    # Create a plot
    plt = plot(features[:, :MA5], label="Moving 5 Day average", title="Moving day averages")
    plot!(features[:, :MA20], label="Moving 20 Day average")

    ns = size(features[:, :MA5], 1)
    # Show legend
    plot!(plt, legend=false)

    return plt

end

function plt_rsi_indicator(features)

    # Create a plot
    plt = plot(features[:, :RSI14], label="RSI 14-day")
    ns = size(features[:, :RSI14], 1)

    # Add dashed lines for thresholds
    plot!(plt, 1:ns, ones(ns, 1)*30, label="Oversold threshold", linestyle=:dash, linecolor=:green)
    plot!(plt, 1:ns, ones(ns, 1)*70, label="Overbought threshold", linestyle=:dash, linecolor=:red)

    # Show legend
    plot!(plt, legend=false)
    return plt

end

function plt_momentum_indicator(features)

    # Create a plot
    plt = plot(features[:, :momentum14], title="Momentum 14-day")
    ns = size(features[:, :RSI14], 1)

    # Add dashed lines for thresholds
    # plot!(plt, 1:ns, ones(ns, 1)*30, label="Oversold threshold", linestyle=:dash, linecolor=:green)
    # plot!(plt, 1:ns, ones(ns, 1)*70, label="Overbought threshold", linestyle=:dash, linecolor=:red)

    # Show legend
    plot!(plt, legend=false)
    return plt

end

export plot_loss_data
function plot_loss_data(losses, loader)
    plt = Plots.plot()
    plot!(plt, losses; xaxis=(:log10, "iteration"),
        yaxis="loss", label="per batch")
    n = length(loader)
    plot!(plt, n:n:length(losses), mean.(Iterators.partition(losses, n)),
        label="epoch mean", dpi=200)
end