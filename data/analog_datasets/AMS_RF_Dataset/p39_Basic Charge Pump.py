from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *
import matplotlib.pyplot as plt
import numpy as np

# Create the circuit
circuit = Circuit('Basic Charge Pump')

# Define power supply
circuit.V('dd', 'Vdd', circuit.gnd, 5@u_V)

# Define clock signals (non-overlapping clocks for charge pump operation)
circuit.PulseVoltageSource('clk1', 'phi1', circuit.gnd, 
                          initial_value=0@u_V, pulsed_value=5@u_V,
                          pulse_width=500@u_ns, period=1@u_us,
                          rise_time=10@u_ns, fall_time=10@u_ns)
circuit.PulseVoltageSource('clk2', 'phi2', circuit.gnd, 
                          initial_value=0@u_V, pulsed_value=5@u_V,
                          pulse_width=500@u_ns, period=1@u_us,
                          rise_time=10@u_ns, fall_time=10@u_ns,
                          delay_time=500@u_ns)  # Phase shifted

# Define MOSFET models
circuit.model('NMOS', 'nmos', 
              level=1,
              kp=120e-6,
              vto=0.7,
              lambda_=0.02,
              w=10e-6,
              l=1e-6)
circuit.model('PMOS', 'pmos', 
              level=1,
              kp=60e-6,
              vto=-0.7,
              lambda_=0.02,
              w=20e-6,
              l=1e-6)

# Charge pump components - corrected architecture
# First stage
circuit.MOSFET('M1', 'node1', 'phi1', circuit.gnd, circuit.gnd, model='NMOS')  # Switching NMOS
circuit.C('C1', 'node1', 'phi2', 10@u_pF)  # Pumping capacitor connected to phi2

# Second stage (diode-connected MOSFET for charge transfer)
circuit.MOSFET('M2', 'Vout', 'node1', 'node1', circuit.gnd, model='NMOS')  # Diode-connected transfer MOSFET
circuit.C('C2', 'Vout', circuit.gnd, 100@u_pF)  # Output storage capacitor

# Output load
circuit.R('load', 'Vout', circuit.gnd, 1@u_MΩ)

# Setup simulation
simulator = circuit.simulator(temperature=25, nominal_temperature=25)
simulator.options(reltol=1e-4, abstol=1e-9, vntol=1e-6)

# Perform transient analysis
try:
    analysis = simulator.transient(
        step_time=10@u_ns, 
        end_time=100@u_us,  # Increased to see the pump effect
        use_initial_condition=True
    )
except Exception as e:
    print(f"Simulation error: {e}")
    # Retry with adjusted parameters if needed
    analysis = simulator.transient(
        step_time=100@u_ns, 
        end_time=20@u_us
    )

# Plot results
plt.figure(figsize=(10, 8))

# Plot clock signals
plt.subplot(3, 1, 1)
plt.plot(analysis.time, analysis['phi1'], label='Phi1')
plt.plot(analysis.time, analysis['phi2'], label='Phi2')
plt.title('Charge Pump Clock Signals')
plt.xlabel('Time [s]')
plt.ylabel('Voltage [V]')
plt.legend()
plt.grid(True)

# Plot intermediate node voltage
plt.subplot(3, 1, 2)
plt.plot(analysis.time, analysis['node1'], label='Node1 Voltage', color='green')
plt.title('Intermediate Node Voltage')
plt.xlabel('Time [s]')
plt.ylabel('Voltage [V]')
plt.legend()
plt.grid(True)

# Plot output voltage
plt.subplot(3, 1, 3)
plt.plot(analysis.time, analysis['Vout'], label='Output Voltage', color='red')
plt.title('Charge Pump Output Voltage')
plt.xlabel('Time [s]')
plt.ylabel('Voltage [V]')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Print final output voltage
final_voltage = analysis['Vout'][-1]
print(f"Final output voltage: {final_voltage}")