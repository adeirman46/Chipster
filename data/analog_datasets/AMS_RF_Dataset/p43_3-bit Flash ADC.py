import math
from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *
from PySpice.Spice.Netlist import SubCircuitFactory
import matplotlib.pyplot as plt
import numpy as np

class Opamp(SubCircuitFactory):
    NAME = ('Opamp')
    NODES = ('Vinp', 'Vinn', 'Vout')
    def __init__(self):
        super().__init__()
        # Define the MOSFET models with higher gain for sharper transitions
        self.model('nmos_model', 'nmos', level=1, kp=200e-6, vto=0.5, lambda_=0.01)
        self.model('pmos_model', 'pmos', level=1, kp=100e-6, vto=-0.5, lambda_=0.01)
        
        # Internal power supply and bias
        self.V('dd_int', 'Vdd_int', self.gnd, 5.0)
        self.V('bias', 'Vbias', self.gnd, 1.5)
        
        # Differential pair with larger sizes for higher transconductance
        self.MOSFET('1', 'Voutp', 'Vinp', 'Source3', 'Source3', model='nmos_model', w=100e-6, l=0.5e-6)
        self.MOSFET('2', 'Vout', 'Vinn', 'Source3', 'Source3', model='nmos_model', w=100e-6, l=0.5e-6)
        
        # Tail current source with larger width for more current
        self.MOSFET('3', 'Source3', 'Vbias', self.gnd, self.gnd, model='nmos_model', w=200e-6, l=1e-6)
        
        # Current mirror load with higher current capability
        self.MOSFET('4', 'Voutp', 'Voutp', 'Vdd_int', 'Vdd_int', model='pmos_model', w=200e-6, l=0.5e-6)
        self.MOSFET('5', 'Vout', 'Voutp', 'Vdd_int', 'Vdd_int', model='pmos_model', w=200e-6, l=0.5e-6)

# Create a 3-bit Flash ADC circuit
circuit = Circuit('3-bit Flash ADC')

# Define power supply
circuit.V('dd', 'Vdd', circuit.gnd, 5@u_V)

# Create precision resistor ladder for reference voltages
# Using smaller, matched resistors for better accuracy
r_ladder = 500@u_Ω  

# Create voltage divider chain from top to bottom
# This creates references at: 4.375V, 3.75V, 3.125V, 2.5V, 1.875V, 1.25V, 0.625V
circuit.R('R_top', 'Vdd', 'Vref7', r_ladder)      # Top resistor
circuit.R('R6', 'Vref7', 'Vref6', r_ladder)       # 4.375V to 3.75V
circuit.R('R5', 'Vref6', 'Vref5', r_ladder)       # 3.75V to 3.125V  
circuit.R('R4', 'Vref5', 'Vref4', r_ladder)       # 3.125V to 2.5V
circuit.R('R3', 'Vref4', 'Vref3', r_ladder)       # 2.5V to 1.875V
circuit.R('R2', 'Vref3', 'Vref2', r_ladder)       # 1.875V to 1.25V
circuit.R('R1', 'Vref2', 'Vref1', r_ladder)       # 1.25V to 0.625V
circuit.R('R_bot', 'Vref1', circuit.gnd, r_ladder) # Bottom resistor

# Add buffer resistors to prevent loading of reference voltages
for i in range(1, 8):
    circuit.R(f'Rbuf{i}', f'Vref{i}', f'Vref{i}_buf', 1@u_Ω)

# Declare the opamp subcircuit
circuit.subcircuit(Opamp())

# Create 7 comparators using the op-amp
# For proper Flash ADC operation:
# - Input goes to non-inverting input (+)
# - Reference goes to inverting input (-)
# - When Vin > Vref, output goes HIGH
# - When Vin < Vref, output goes LOW

for i in range(1, 8):
    # Create comparator: Vin(+) compared with Vref_i(-)
    circuit.X(f'cmp{i}', 'Opamp', 'Vin', f'Vref{i}_buf', f'Comp_out_{i}')
    
    # Add pull-up resistors to ensure proper HIGH levels
    circuit.R(f'Rpull{i}', f'Comp_out_{i}', 'Vdd', 10@u_kΩ)

# Input voltage source
circuit.V('input', 'Vin', circuit.gnd, 2.5@u_V)

# Setup simulation with improved convergence
simulator = circuit.simulator(temperature=25, nominal_temperature=25)
simulator.options(
    reltol=1e-4,
    abstol=1e-10,
    vntol=1e-6,
    method='gear',
    maxiter=200,
    gmin=1e-15,
    pivrel=1e-3
)

try:
    # Perform DC analysis with finer resolution
    analysis = simulator.dc(Vinput=slice(0, 5, 0.02))
    
    # Extract results
    input_voltage = np.array(analysis.Vin)
    
    # Print actual reference voltages
    print("Actual Reference Voltages:")
    print("=" * 30)
    ref_voltages = {}
    for i in range(1, 8):
        vref_actual = float(analysis[f'Vref{i}'][0])
        vref_expected = 5.0 * i / 8.0
        ref_voltages[i] = vref_actual
        print(f"Vref{i}: {vref_actual:.3f}V (Expected: {vref_expected:.3f}V)")
    
    # Create comprehensive plots
    plt.figure(figsize=(16, 12))
    
    # Plot 1: Reference voltage verification
    plt.subplot(2, 2, 1)
    ref_values = [ref_voltages[i] for i in range(1, 8)]
    expected_values = [5.0 * i / 8.0 for i in range(1, 8)]
    x_pos = range(1, 8)
    
    plt.bar([x - 0.2 for x in x_pos], ref_values, 0.4, label='Actual', alpha=0.7)
    plt.bar([x + 0.2 for x in x_pos], expected_values, 0.4, label='Expected', alpha=0.7)
    plt.xlabel('Reference Number')
    plt.ylabel('Voltage (V)')
    plt.title('Reference Voltage Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Comparator outputs (analog)
    plt.subplot(2, 2, 2)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    
    for i in range(1, 8):
        comp_out = np.array(analysis[f'Comp_out_{i}'])
        plt.plot(input_voltage, comp_out, color=colors[i-1], 
                linewidth=2, label=f'Comp {i} (Vref={ref_voltages[i]:.2f}V)')
    
    plt.title('Flash ADC - Comparator Analog Outputs')
    plt.xlabel('Input Voltage (V)')
    plt.ylabel('Comparator Output Voltage (V)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Digital thermometer code
    plt.subplot(2, 2, 3)
    
    threshold = 2.5  # Digital threshold voltage
    digital_outputs = []
    
    for i in range(1, 8):
        comp_out = np.array(analysis[f'Comp_out_{i}'])
        digital_out = (comp_out > threshold).astype(int)
        digital_outputs.append(digital_out)
        
        # Plot with offset for visibility
        plt.plot(input_voltage, digital_out * 0.8 + i - 0.5, 
                color=colors[i-1], linewidth=3, label=f'Comp {i}')
    
    plt.title('Flash ADC - Digital Thermometer Code')
    plt.xlabel('Input Voltage (V)')
    plt.ylabel('Comparator Number')
    plt.ylim(0, 8)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot 4: 3-bit binary output simulation
    plt.subplot(2, 2, 4)
    
    # Convert thermometer code to binary
    binary_codes = []
    for j in range(len(input_voltage)):
        # Count number of HIGH comparators
        high_count = sum([digital_outputs[i][j] for i in range(7)])
        binary_codes.append(high_count)
    
    plt.plot(input_voltage, binary_codes, 'ko-', linewidth=2, markersize=3)
    plt.title('Flash ADC - 3-bit Digital Output')
    plt.xlabel('Input Voltage (V)')
    plt.ylabel('Digital Code (0-7)')
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.5, 7.5)
    
    # Add step annotations
    for code in range(8):
        plt.axhline(y=code, color='gray', linestyle='--', alpha=0.3)
        voltage_range = f"{code*5/8:.2f}V-{(code+1)*5/8:.2f}V"
        if code < 7:
            plt.text(0.1, code + 0.3, f"Code {code}", fontsize=8)
    
    plt.tight_layout()
    plt.savefig('flash_adc_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Performance analysis
    print(f"\nFlash ADC Performance Analysis:")
    print("=" * 50)
    
    transition_points = []
    for i in range(1, 8):
        comp_out = np.array(analysis[f'Comp_out_{i}'])
        
        # Find transition point where output crosses threshold
        try:
            # Find where output transitions from low to high or high to low
            diff_out = np.diff(comp_out)
            max_change_idx = np.argmax(np.abs(diff_out))
            transition_vin = input_voltage[max_change_idx]
            
            expected_vref = ref_voltages[i]
            error = abs(transition_vin - expected_vref)
            
            transition_points.append(transition_vin)
            print(f"Comparator {i}: Transitions at {transition_vin:.3f}V "
                  f"(Vref: {expected_vref:.3f}V, Error: {error:.3f}V)")
                  
        except:
            print(f"Comparator {i}: Could not determine clear transition point")
    
    # Overall ADC metrics
    if len(transition_points) > 1:
        step_sizes = np.diff(sorted(transition_points))
        avg_step = np.mean(step_sizes)
        step_variation = np.std(step_sizes)
        
        print(f"\nADC Metrics:")
        print(f"Average step size: {avg_step:.3f}V")
        print(f"Step size variation (std): {step_variation:.3f}V")
        print(f"Theoretical step size: {5.0/8:.3f}V")
        print(f"Resolution: 3 bits ({2**3} levels)")
        
        if step_variation < 0.1:  # Arbitrary threshold for "good" performance
            print("✓ Flash ADC is functioning correctly!")
        else:
            print("⚠ Large step size variation detected - check component matching")
    
except Exception as e:
    print(f"Simulation failed: {e}")
    import traceback
    traceback.print_exc()
    print("\nTroubleshooting suggestions:")
    print("1. The op-amp model may be too complex for convergence")
    print("2. Try reducing resistor values or increasing capacitive loading")
    print("3. Consider using ideal voltage sources for references initially")