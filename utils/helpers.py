def bar_chart_nums(bar_chart, axe, nums_to_show):
  for i, bar in enumerate(bar_chart):
    yval = bar.get_height()
    axe.text(bar.get_x() + bar.get_width() / 2, yval + 0.02 if yval > 0 else yval - 0.08, f"{nums_to_show[i]:.2f}", 
             ha='center', va='bottom', fontsize=8)