from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *
import matplotlib.pyplot as plt
import numpy as np

# Create a simpler, more robust bandgap reference circuit
circuit = Circuit('Bandgap Reference Circuit')

# Define power supply
circuit.V('dd', 'Vdd', circuit.gnd, 3.3@u_V)

# Define bipolar transistors with different areas (8:1 ratio)
circuit.BJT('Q1', 'Q1_collector', 'Q1_base', circuit.gnd, model='NPN', area=1)
circuit.BJT('Q2', 'Q2_collector', 'Q2_base', circuit.gnd, model='NPN', area=8)

# Add small resistors to collectors for better convergence
circuit.R('R1', 'Vdd', 'Q1_collector', 10@u_kΩ)
circuit.R('R2', 'Vdd', 'Q2_collector', 10@u_kΩ)

# Add base resistors
circuit.R('R3', 'Q1_base', 'Q1_collector', 5@u_kΩ)
circuit.R('R4', 'Q2_base', 'Q2_collector', 5@u_kΩ)

# Add a simple current mirror to bias the transistors
circuit.BJT('Q3', 'Q3_collector', 'Q3_collector', circuit.gnd, model='NPN', area=1)  # Diode-connected
circuit.R('R5', 'Vdd', 'Q3_collector', 10@u_kΩ)

# Connect the current mirror to the bandgap core
circuit.R('R6', 'Q3_collector', 'Q1_base', 5@u_kΩ)
circuit.R('R7', 'Q3_collector', 'Q2_base', 5@u_kΩ)

# Add a PTAT resistor between the collectors
circuit.R('Rptat', 'Q1_collector', 'Q2_collector', 2@u_kΩ)

# Output stage - simple voltage follower
circuit.BJT('Q4', 'Vout', 'Q2_collector', circuit.gnd, model='NPN', area=1)
circuit.R('Rout', 'Vdd', 'Vout', 5@u_kΩ)

# Define device models with proper parameters
circuit.model('NPN', 'npn',
              is_=1e-16,
              bf=100,
              br=1,
              vaf=50,
              ikf=0.1,
              ise=1e-15,
              ne=1.5,
              rc=10)

# Setup simulation with convergence helpers
simulator = circuit.simulator(temperature=25, nominal_temperature=25)
simulator.options(
    reltol=1e-3,
    abstol=1e-9,
    vntol=1e-6,
    gmin=1e-12,
    method='gear',
    itl1=1000,
    itl2=1000,
    itl4=1000,
    srcsteps=100,
    pivtol=1e-12,
    pivrel=1e-3
)

print("Testing Bandgap Reference Circuit...")

# Operating point analysis
try:
    analysis_op = simulator.operating_point()
    vout = float(analysis_op['Vout'])
    print(f"Operating Point Analysis: Vout = {vout:.6f} V")
    
    # Test if circuit is working
    if 1.1 <= vout <= 1.3:  # Typical bandgap voltage range
        print("✓ PASS: Circuit is generating a proper reference voltage")
    else:
        print("✗ FAIL: Circuit is not generating a proper reference voltage")
        
except Exception as e:
    print(f"✗ FAIL: Operating point analysis failed: {e}")
    # Try a DC analysis instead
    try:
        analysis_dc = simulator.dc(Vdd=slice(0, 3.3, 0.1))
        vout = float(analysis_dc['Vout'][-1])  # Get the last value
        print(f"DC Analysis: Vout at 3.3V = {vout:.6f} V")
    except Exception as e2:
        print(f"DC analysis also failed: {e2}")

# DC analysis - temperature sweep
print("\nTemperature Stability Test:")
temperatures = np.linspace(-40, 125, 10)
vout_values = []
success_count = 0

for temp in temperatures:
    try:
        # Create a new simulator for each temperature
        temp_simulator = circuit.simulator(temperature=temp, nominal_temperature=25)
        temp_simulator.options(
            reltol=1e-3,
            abstol=1e-9,
            vntol=1e-6,
            gmin=1e-12,
            itl1=1000,
            itl2=1000,
            itl4=1000
        )
        analysis = temp_simulator.operating_point()
        vout_val = float(analysis['Vout'])
        vout_values.append(vout_val)
        print(f"Temperature {temp}°C: Vout = {vout_val:.6f} V")
        success_count += 1
    except Exception as e:
        print(f"Temperature {temp}°C: Failed - {e}")
        vout_values.append(np.nan)

# Plot the temperature stability results
if success_count > 0:
    plt.figure(figsize=(10, 6))
    
    # Filter out failed simulations
    valid_temps = []
    valid_vouts = []
    for i, (temp, vout) in enumerate(zip(temperatures, vout_values)):
        if not np.isnan(vout):
            valid_temps.append(temp)
            valid_vouts.append(vout)
    
    if len(valid_temps) > 1:
        plt.plot(valid_temps, valid_vouts, 'o-', linewidth=2, markersize=8)
        plt.xlabel('Temperature (°C)')
        plt.ylabel('Output Voltage (V)')
        plt.title('Bandgap Reference Voltage vs Temperature')
        plt.grid(True, alpha=0.3)
        
        # Add voltage range indicators
        if len(valid_vouts) > 0:
            avg_voltage = np.mean(valid_vouts)
            plt.axhline(y=avg_voltage, color='r', linestyle='--', alpha=0.7, label=f'Average: {avg_voltage:.3f} V')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig('bandgap_temperature_stability.png', dpi=150)
        plt.show()
        
        # Calculate and display statistics
        vout_range = max(valid_vouts) - min(valid_vouts)
        vout_std = np.std(valid_vouts)
        print(f"\nTemperature Stability Statistics:")
        print(f"  Voltage range: {vout_range*1000:.2f} mV")
        print(f"  Standard deviation: {vout_std*1000:.2f} mV")
        
        # Test temperature stability
        if vout_range < 0.1:  # Less than 100mV variation
            print(f"✓ PASS: Good temperature stability (ΔV = {vout_range*1000:.2f} mV)")
        else:
            print(f"✗ FAIL: Poor temperature stability (ΔV = {vout_range*1000:.2f} mV)")
    else:
        print("✗ FAIL: Insufficient data for plotting")
else:
    print("✗ FAIL: Insufficient data for temperature stability test")

print("\nBandgap Reference Test Complete")