"""
This macro finds the multiplicity % cut based on the centrality % cut for 5.02TeV Pb+Pb collisions using the data given in PRL 116, 222302 (2016) for midrapidity range.

"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

centrality_midpoints = np.array([1.25, 3.75, 6.25, 8.75, 15, 25, 35, 45, 55, 65, 75])
multiplicity = np.array([2035, 1850, 1666, 1505, 1180, 786, 512, 318, 183, 96.3, 44.9])
multiplicity_errors = np.array([52, 55, 48, 44, 31, 20, 15, 12, 8, 5.8, 3.4])

def exponential_fit(x, A, n):
    return A * np.exp(-(x**n))

def exponential_fit_extended(x, A, B, n):
    return A * np.exp(-((x/B)**n))

print("Centrality vs Multiplicity Analysis for Pb-Pb at 5.02 TeV")
print("="*60)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

ax1.errorbar(centrality_midpoints, multiplicity, yerr=multiplicity_errors, 
             fmt='ro', markersize=8, capsize=5, capthick=2, label='ALICE Data')
ax1.set_xlabel('Centrality (%)', fontsize=12)
ax1.set_ylabel('dN_ch/dη', fontsize=12)
ax1.set_title('Charged Particle Multiplicity vs Centrality\nPb-Pb at √s_NN = 5.02 TeV', fontsize=14)
ax1.grid(True, alpha=0.3)

try:
    popt1, pcov1 = curve_fit(exponential_fit, centrality_midpoints, multiplicity, 
                            p0=[2500, 0.5], maxfev=5000)
    A1, n1 = popt1
    
    popt2, pcov2 = curve_fit(exponential_fit_extended, centrality_midpoints, multiplicity, 
                            p0=[2500, 10, 1], maxfev=5000)
    A2, B2, n2 = popt2
    
    x_smooth = np.linspace(0.1, 80, 1000)
    y_fit1 = exponential_fit(x_smooth, A1, n1)
    y_fit2 = exponential_fit_extended(x_smooth, A2, B2, n2)
    
    ss_res1 = np.sum((multiplicity - exponential_fit(centrality_midpoints, A1, n1)) ** 2)
    ss_tot = np.sum((multiplicity - np.mean(multiplicity)) ** 2)
    r_squared1 = 1 - (ss_res1 / ss_tot)
    
    ss_res2 = np.sum((multiplicity - exponential_fit_extended(centrality_midpoints, A2, B2, n2)) ** 2)
    r_squared2 = 1 - (ss_res2 / ss_tot)
    
    ax1.plot(x_smooth, y_fit1, 'b--', linewidth=2, 
             label=f'Fit 1: A·exp(-x^n)\nA={A1:.0f}, n={n1:.3f}\nR²={r_squared1:.4f}')
    ax1.plot(x_smooth, y_fit2, 'g-', linewidth=2, 
             label=f'Fit 2: A·exp(-(x/B)^n)\nA={A2:.0f}, B={B2:.2f}, n={n2:.3f}\nR²={r_squared2:.4f}')
    
    print(f"Fit 1 - Simple exponential A*exp(-x^n):")
    print(f"  A = {A1:.2f}")
    print(f"  n = {n1:.4f}")
    print(f"  R² = {r_squared1:.4f}")
    print()
    print(f"Fit 2 - Extended exponential A*exp(-(x/B)^n):")
    print(f"  A = {A2:.2f}")
    print(f"  B = {B2:.4f}")
    print(f"  n = {n2:.4f}")
    print(f"  R² = {r_squared2:.4f}")
    
    if r_squared2 > r_squared1:
        best_fit_func = lambda x: exponential_fit_extended(x, A2, B2, n2)
        print(f"\nBest fit: Extended exponential (R² = {r_squared2:.4f})")
        fit_params = f"A={A2:.0f}, B={B2:.2f}, n={n2:.3f}"
    else:
        best_fit_func = lambda x: exponential_fit(x, A1, n1)
        print(f"\nBest fit: Simple exponential (R² = {r_squared1:.4f})")
        fit_params = f"A={A1:.0f}, n={n1:.3f}"
    
    mult_at_0 = best_fit_func(0.01)
    mult_at_0_5 = best_fit_func(0.5)
    
    absolute_change = mult_at_0 - mult_at_0_5
    percent_change = (absolute_change / mult_at_0) * 100
    
    print(f"\nParticle Change Analysis (0% to 0.5% centrality):")
    print("="*50)
    print(f"Multiplicity at ~0%: {mult_at_0:.1f}")
    print(f"Multiplicity at 0.5%: {mult_at_0_5:.1f}")
    print(f"Absolute change: {absolute_change:.1f}")
    print(f"Percentage change: {percent_change:.2f}%")
    
    ax1.axvline(x=0.5, color='red', linestyle=':', alpha=0.7, label='0.5% centrality')
    ax1.axvspan(0, 0.5, alpha=0.1, color='red', label='0-0.5% region')
    
    textstr = f'0-0.5% Centrality Range:\nΔ(dN_ch/dη) ≈ {absolute_change:.0f}\nRelative change ≈ {percent_change:.1f}%'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax1.text(0.65, 0.95, textstr, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    ax1.set_xlim(0, 80)
    ax1.set_ylim(0, 2200)
    ax1.legend(loc='upper right')
    
    x_zoom = np.linspace(0.01, 10, 100)
    y_zoom = best_fit_func(x_zoom)
    ax2.plot(x_zoom, y_zoom, 'g-', linewidth=3, label=f'Best fit: {fit_params}')
    
    central_mask = centrality_midpoints <= 10
    ax2.errorbar(centrality_midpoints[central_mask], multiplicity[central_mask], 
                 yerr=multiplicity_errors[central_mask], 
                 fmt='ro', markersize=10, capsize=5, capthick=2, label='ALICE Data')
    
    ax2.axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='0.5% centrality')
    ax2.axvspan(0, 0.5, alpha=0.2, color='red', label='0-0.5% region')
    
    ax2.set_xlabel('Centrality (%)', fontsize=12)
    ax2.set_ylabel('dN_ch/dη', fontsize=12)
    ax2.set_title('Ultra-Central Region: 0-10% Centrality\nPb-Pb at √s_NN = 5.02 TeV', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
except Exception as e:
    print(f"Error in fitting: {e}")
    print("Proceeding with interpolation method...")
    
    interp_func = interp1d(centrality_midpoints, multiplicity, kind='cubic', 
                          fill_value='extrapolate')
    
    mult_at_0_5_interp = interp_func(0.5)
    mult_at_0_interp = interp_func(0.01)
    
    change_interp = mult_at_0_interp - mult_at_0_5_interp
    percent_change_interp = (change_interp / mult_at_0_interp) * 100
    
    print(f"Using interpolation:")
    print(f"Multiplicity at ~0%: {mult_at_0_interp:.1f}")
    print(f"Multiplicity at 0.5%: {mult_at_0_5_interp:.1f}")
    print(f"Change: {change_interp:.1f} ({percent_change_interp:.2f}%)")

plt.tight_layout()
plt.savefig('centrality_mult.png', dpi=300, bbox_inches='tight')
print(f"\nPlot saved as 'centrality_mult.png'")
plt.close()

print("\nNote: The analysis shows the expected particle multiplicity change")
print("in the ultra-central 0-0.5% region based on the fitted function.")
