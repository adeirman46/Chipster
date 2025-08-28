from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *
import matplotlib.pyplot as plt
import numpy as np

# Create the circuit
circuit = Circuit('Operational Transconductance Amplifier (OTA)')

# Define power supplies
circuit.V('dd', 'Vdd', circuit.gnd, 5@u_V)  # Positive supply
circuit.V('ss', 'Vss', circuit.gnd, -5@u_V)  # Negative supply

# Define bias current source
circuit.I('bias', 'Vdd', 'bias_node', 50@u_uA)  # Bias current

# Define input signals (differential)
circuit.SinusoidalVoltageSource('in_p', 'in_p', circuit.gnd, 
                               dc_offset=0@u_V, amplitude=0.01@u_V, frequency=1@u_kHz)
circuit.V('in_n', 'in_n', circuit.gnd, 0@u_V)  # DC reference

# Define MOSFET models with proper parameters
circuit.model('NMOS', 'nmos', 
              level=1,
              kp=120e-6,
              vto=0.7,
              lambda_=0.02,
              gamma=0.5,
              phi=0.7)

circuit.model('PMOS', 'pmos', 
              level=1,
              kp=40e-6,
              vto=-0.7,
              lambda_=0.02,
              gamma=0.5,
              phi=0.7)

# Differential pair (NMOS transistors)
circuit.MOSFET(1, 'drain1', 'in_p', 'tail', circuit.gnd, model='NMOS', w=50e-6, l=1e-6)
circuit.MOSFET(2, 'drain2', 'in_n', 'tail', circuit.gnd, model='NMOS', w=50e-6, l=1e-6)

# Tail current source (NMOS current mirror)
circuit.MOSFET(3, 'tail', 'bias_node', 'Vss', 'Vss', model='NMOS', w=20e-6, l=1e-6)
circuit.MOSFET(4, 'bias_node', 'bias_node', 'Vss', 'Vss', model='NMOS', w=20e-6, l=1e-6)

# Current mirror load (PMOS transistors)
circuit.MOSFET(5, 'drain1', 'drain1', 'Vdd', 'Vdd', model='PMOS', w=100e-6, l=1e-6)
circuit.MOSFET(6, 'drain2', 'drain1', 'Vdd', 'Vdd', model='PMOS', w=100e-6, l=1e-6)

# Output stage
circuit.MOSFET(7, 'output', 'drain2', 'Vss', 'Vss', model='NMOS', w=50e-6, l=1e-6)
circuit.MOSFET(8, 'output', 'bias_node', 'Vdd', 'Vdd', model='PMOS', w=100e-6, l=1e-6)

# Add compensation for stability
circuit.C('comp', 'drain2', 'output', 2@u_pF)  # Miller compensation capacitor
circuit.R('comp_res', 'drain2', 'comp_node', 1@u_kΩ)  # Compensation resistor
circuit.C('comp2', 'comp_node', 'output', 2@u_pF)  # Second compensation capacitor

# Add a load capacitor
circuit.C('load', 'output', circuit.gnd, 10@u_pF)

# Add a small resistor in series with the load to prevent oscillations
circuit.R('series', 'output', 'out_node', 100@u_Ω)
circuit.C('load2', 'out_node', circuit.gnd, 10@u_pF)

# Setup simulation with more conservative options for stability
simulator = circuit.simulator(temperature=25, nominal_temperature=25)
simulator.options(
    reltol=1e-6, 
    abstol=1e-12, 
    vntol=1e-6,
    method='gear',  # More stable integration method
    itl1=100,       # Increase DC iteration limit
    itl2=50,        # Increase transient iteration limit
    itl4=20,        # Increase transient timepoint iteration limit
    pivotrel=1e-3,  # Better pivot relative tolerance
    pivottol=1e-6   # Better pivot absolute tolerance
)

print("Circuit netlist:")
print(circuit)

# Run operating point analysis
print("\nOperating Point Analysis:")
try:
    dc_analysis = simulator.operating_point()
    # Convert to regular Python values
    for node_name in dc_analysis.nodes.keys():
        node_value = dc_analysis[node_name]
        if hasattr(node_value, 'as_ndarray'):
            node_value = node_value.as_ndarray()[0]
        print(f"{node_name}: {node_value:.6f} V")
except Exception as e:
    print(f"Operating point analysis failed: {e}")

# Run transient analysis with smaller steps for stability
print("\nRunning transient analysis...")
try:
    transient_analysis = simulator.transient(
        step_time=0.1@u_us,  # Smaller step time
        end_time=2@u_ms
    )
except Exception as e:
    print(f"Transient analysis failed: {e}")
    transient_analysis = None

# Run AC analysis
print("\nRunning AC analysis...")
try:
    ac_analysis = simulator.ac(
        start_frequency=1@u_Hz,
        stop_frequency=100@u_MHz,
        number_of_points=200,
        variation='dec'
    )
except Exception as e:
    print(f"AC analysis failed: {e}")
    ac_analysis = None

# Plot results if analyses were successful
if transient_analysis is not None:
    plt.figure(figsize=(12, 8))

    # Convert to numpy arrays
    time = np.array(transient_analysis.time)
    in_p = np.array(transient_analysis['in_p'])
    output = np.array(transient_analysis['out_node'])  # Use the node after series resistor
    
    # Transient analysis plot
    plt.subplot(2, 2, 1)
    plt.plot(time, in_p, label='Input+')
    plt.plot(time, output, label='Output')
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (V)')
    plt.title('Transient Response (Stabilized)')
    plt.legend()
    plt.grid(True)

if ac_analysis is not None:
    # Convert to numpy arrays
    frequency = np.array(ac_analysis.frequency)
    output_ac = np.array(ac_analysis['out_node'])  # Use the node after series resistor
    
    # AC analysis plot - magnitude
    plt.subplot(2, 2, 2)
    gain = np.abs(output_ac)
    plt.semilogx(frequency, 20*np.log10(np.where(gain > 0, gain, 1e-12)))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain (dB)')
    plt.title('AC Response - Magnitude (Stabilized)')
    plt.grid(True)

    # AC analysis plot - phase
    plt.subplot(2, 2, 3)
    plt.semilogx(frequency, np.angle(output_ac, deg=True))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Phase (degrees)')
    plt.title('AC Response - Phase (Stabilized)')
    plt.grid(True)

# DC transfer characteristic
try:
    dc_sweep = simulator.dc(Vin_p=slice(-0.1, 0.1, 0.005))
    plt.subplot(2, 2, 4)
    plt.plot(dc_sweep.Vin_p, dc_sweep.out_node)  # Use the node after series resistor
    plt.xlabel('Differential Input Voltage (V)')
    plt.ylabel('Output Voltage (V)')
    plt.title('DC Transfer Characteristic (Stabilized)')
    plt.grid(True)
except Exception as e:
    print(f"DC sweep failed: {e}")

plt.tight_layout()
plt.show()

# Calculate and print performance metrics
print("\nPerformance Metrics:")
    
# DC gain calculation
try:
    if ac_analysis is not None:
        # Get the low-frequency gain (first point in AC analysis)
        low_freq_gain = np.abs(output_ac[0])
        if low_freq_gain > 0:
            print(f"DC Gain: {low_freq_gain:.2f} ({20*np.log10(low_freq_gain):.2f} dB)")
        else:
            print("DC Gain: 0.00 (-inf dB)")
except Exception as e:
    print(f"Could not calculate DC gain: {e}")
        
# Phase margin calculation
try:
    if ac_analysis is not None:
        # Find unity gain frequency
        unity_gain_idx = np.where(gain <= 1)[0]
        if len(unity_gain_idx) > 0:
            ugf = frequency[unity_gain_idx[0]]
            phase_at_ugf = np.angle(output_ac[unity_gain_idx[0]], deg=True)
            phase_margin = 180 + phase_at_ugf
            print(f"Unity Gain Frequency: {ugf:.2e} Hz")
            print(f"Phase Margin: {phase_margin:.2f}°")
            
            # Check if phase margin is sufficient for stability
            if phase_margin > 45:
                print("Phase margin is sufficient for stability (>45°)")
            else:
                print("WARNING: Phase margin may be insufficient for stability")
        else:
            print("Could not find unity gain frequency")
except Exception as e:
    print(f"Could not calculate phase margin: {e}")

# Additional metrics from operating point
try:
    if 'dc_analysis' in locals():
        output_voltage = dc_analysis['out_node']
        if hasattr(output_voltage, 'as_ndarray'):
            output_voltage = output_voltage.as_ndarray()[0]
        print(f"Output DC voltage: {output_voltage:.3f} V")
        
        # Calculate approximate power consumption
        total_current = 100e-6  # 100μA
        power = 10 * total_current  # 10V total supply * current
        print(f"Approximate power consumption: {power*1e6:.2f} μW")
        
        # Calculate output swing range
        max_output = 4.0  # V
        min_output = -4.0  # V
        output_swing = max_output - min_output
        print(f"Estimated output swing: {output_swing:.1f} V")
except Exception as e:
    print(f"Could not calculate additional metrics: {e}")

# Check for stability in transient response
if transient_analysis is not None:
    output_signal = np.array(transient_analysis['out_node'])
    # Check if the output is oscillating by looking for significant variations
    std_dev = np.std(output_signal)
    mean_val = np.mean(output_signal)
    
    if std_dev > 0.1 * abs(mean_val):  # If standard deviation is more than 10% of mean
        print("WARNING: Output shows significant oscillation")
        print(f"Output standard deviation: {std_dev:.4f} V")
    else:
        print("Output appears stable")
        print(f"Output standard deviation: {std_dev:.6f} V")