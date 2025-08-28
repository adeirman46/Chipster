from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Create a new circuit
circuit = Circuit('4:1 CMOS Multiplexer')

# Define power supply
circuit.V('dd', 'Vdd', circuit.gnd, 5@u_V)

# Define input signals as pulse sources with different patterns
circuit.PulseVoltageSource('in0', 'in0_node', circuit.gnd,
                          initial_value=0@u_V, pulsed_value=5@u_V,
                          pulse_width=400@u_ns, period=800@u_ns,
                          delay_time=0@u_ns, rise_time=10@u_ns, fall_time=10@u_ns)
circuit.PulseVoltageSource('in1', 'in1_node', circuit.gnd,
                          initial_value=0@u_V, pulsed_value=5@u_V,
                          pulse_width=200@u_ns, period=400@u_ns,
                          delay_time=0@u_ns, rise_time=10@u_ns, fall_time=10@u_ns)
circuit.PulseVoltageSource('in2', 'in2_node', circuit.gnd,
                          initial_value=0@u_V, pulsed_value=5@u_V,
                          pulse_width=100@u_ns, period=200@u_ns,
                          delay_time=0@u_ns, rise_time=10@u_ns, fall_time=10@u_ns)
circuit.PulseVoltageSource('in3', 'in3_node', circuit.gnd,
                          initial_value=0@u_V, pulsed_value=5@u_V,
                          pulse_width=50@u_ns, period=100@u_ns,
                          delay_time=0@u_ns, rise_time=10@u_ns, fall_time=10@u_ns)

# Define select signals (S0 and S1) with slower transitions
circuit.PulseVoltageSource('S0', 'S0_node', circuit.gnd,
                          initial_value=0@u_V, pulsed_value=5@u_V,
                          pulse_width=1000@u_ns, period=2000@u_ns,)
circuit.PulseVoltageSource('S1', 'S1_node', circuit.gnd,
                          initial_value=0@u_V, pulsed_value=5@u_V,
                          pulse_width=2000@u_ns, period=4000@u_ns,)

# Define MOSFET models
circuit.model('NMOS', 'nmos', 
              level=1, kp=120e-6, vto=0.7, lambda_=0.02, 
              w=10e-6, l=1e-6)
circuit.model('PMOS', 'pmos', 
              level=1, kp=60e-6, vto=-0.7, lambda_=0.02, 
              w=20e-6, l=1e-6)

# Generate complementary select signals using inverters
# Inverter for S0
circuit.MOSFET(1, 'S0_bar', 'S0_node', circuit.gnd, circuit.gnd, model='NMOS')
circuit.MOSFET(2, 'S0_bar', 'S0_node', 'Vdd', 'Vdd', model='PMOS')

# Inverter for S1
circuit.MOSFET(3, 'S1_bar', 'S1_node', circuit.gnd, circuit.gnd, model='NMOS')
circuit.MOSFET(4, 'S1_bar', 'S1_node', 'Vdd', 'Vdd', model='PMOS')

# Implement the 4:1 multiplexer using a hierarchical approach
# First level: Two 2:1 multiplexers controlled by S0
# Second level: 2:1 multiplexer controlled by S1

# First 2:1 mux (inputs 0 and 1, controlled by S0)
circuit.MOSFET(5, 'in0_node', 'S0_bar', 'mux1_out', circuit.gnd, model='NMOS')
circuit.MOSFET(6, 'in0_node', 'S0_node', 'mux1_out', 'Vdd', model='PMOS')
circuit.MOSFET(7, 'in1_node', 'S0_node', 'mux1_out', circuit.gnd, model='NMOS')
circuit.MOSFET(8, 'in1_node', 'S0_bar', 'mux1_out', 'Vdd', model='PMOS')

# Second 2:1 mux (inputs 2 and 3, controlled by S0)
circuit.MOSFET(9, 'in2_node', 'S0_bar', 'mux2_out', circuit.gnd, model='NMOS')
circuit.MOSFET(10, 'in2_node', 'S0_node', 'mux2_out', 'Vdd', model='PMOS')
circuit.MOSFET(11, 'in3_node', 'S0_node', 'mux2_out', circuit.gnd, model='NMOS')
circuit.MOSFET(12, 'in3_node', 'S0_bar', 'mux2_out', 'Vdd', model='PMOS')

# Final 2:1 mux (outputs of first two muxes, controlled by S1)
circuit.MOSFET(13, 'mux1_out', 'S1_bar', 'output', circuit.gnd, model='NMOS')
circuit.MOSFET(14, 'mux1_out', 'S1_node', 'output', 'Vdd', model='PMOS')
circuit.MOSFET(15, 'mux2_out', 'S1_node', 'output', circuit.gnd, model='NMOS')
circuit.MOSFET(16, 'mux2_out', 'S1_bar', 'output', 'Vdd', model='PMOS')

# Add a load resistor and capacitor at the output
circuit.R('load', 'output', circuit.gnd, 10@u_kΩ)
circuit.C('out_cap', 'output', circuit.gnd, 100e-15@u_F)

# Setup simulation
simulator = circuit.simulator(temperature=25, nominal_temperature=25)

# Perform transient analysis with a longer duration to see all combinations
analysis = simulator.transient(step_time=10@u_ns, end_time=4000@u_ns)

# Convert analysis time to nanoseconds for easier interpretation
time_ns = np.array(analysis.time) * 1e9

# Plot input signals in separate figures
plt.figure(figsize=(10, 8))

plt.subplot(4, 1, 1)
plt.plot(time_ns, analysis['in0_node'])
plt.title('Input 0 Signal')
plt.xlabel('Time [ns]')
plt.ylabel('Voltage [V]')
plt.grid()

plt.subplot(4, 1, 2)
plt.plot(time_ns, analysis['in1_node'])
plt.title('Input 1 Signal')
plt.xlabel('Time [ns]')
plt.ylabel('Voltage [V]')
plt.grid()

plt.subplot(4, 1, 3)
plt.plot(time_ns, analysis['in2_node'])
plt.title('Input 2 Signal')
plt.xlabel('Time [ns]')
plt.ylabel('Voltage [V]')
plt.grid()

plt.subplot(4, 1, 4)
plt.plot(time_ns, analysis['in3_node'])
plt.title('Input 3 Signal')
plt.xlabel('Time [ns]')
plt.ylabel('Voltage [V]')
plt.grid()

plt.tight_layout()
plt.show()

# Plot select signals
plt.figure(figsize=(10, 6))
plt.plot(time_ns, analysis['S0_node'], label='Select S0')
plt.plot(time_ns, analysis['S1_node'], label='Select S1')
plt.title('Select Signals')
plt.xlabel('Time [ns]')
plt.ylabel('Voltage [V]')
plt.legend()
plt.grid()
plt.show()

# Plot output signal
plt.figure(figsize=(10, 6))
plt.plot(time_ns, analysis['output'], label='Output')
plt.title('Multiplexer Output')
plt.xlabel('Time [ns]')
plt.ylabel('Voltage [V]')
plt.legend()
plt.grid()
plt.show()