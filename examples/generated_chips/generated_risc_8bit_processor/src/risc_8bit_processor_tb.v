/**
 * Testbench for the 8-bit, 5-stage pipelined RISC processor.
 *
 * This testbench performs the following steps:
 * 1. Instantiates the processor DUT.
 * 2. Generates a clock signal.
 * 3. Applies a reset sequence to initialize the processor.
 * 4. Allows the pre-loaded program within the DUT to run to completion.
 * 5. Waits for the processor's 'halted' signal to be asserted.
 * 6. Includes a timeout mechanism to prevent the simulation from running indefinitely.
 * 7. Once halted, it performs self-checking by verifying the final state of
 *    key registers and a specific data memory location against expected values.
 * 8. Prints a clear PASS/FAIL message and terminates the simulation.
 */
 `include "risc_8bit_defines.vh"
module risc_8bit_processor_tb;

    // Include processor definitions. This is necessary for hierarchical access
    // to DUT internal signals and must be placed inside the module.

    // Testbench parameters
    parameter CLK_PERIOD = 10; // Clock period in ns
    parameter TIMEOUT_CYCLES = 200; // Max cycles to wait for halt

    // Testbench signals
    reg clk;
    reg rst;

    // Variable for tracking simulation time. Moved to module scope for Verilog-2001 compatibility.
    integer cycle_counter;

    // Instantiate the Device Under Test (DUT)
    risc_8bit_processor dut (
        .clk(clk),
        .rst(rst)
    );

    // Clock generator
    initial begin
        clk = 1'b0;
        forever #(CLK_PERIOD / 2) clk = ~clk;
    end

    // Main test sequence and self-checking logic
    initial begin
        // Required for waveform generation
        $dumpfile("design.vcd");
        $dumpvars(0, risc_8bit_processor_tb);

        $display("Starting RISC Processor Testbench...");

        // 1. Apply reset
        rst = 1'b1;
        $display("[%0t] Asserting reset.", $time);
        #(2 * CLK_PERIOD);
        rst = 1'b0;
        $display("[%0t] De-asserting reset. Processor running.", $time);

        // 2. Wait for the program to complete (or timeout)
        cycle_counter = 0; // Initialization
        // Use direct hierarchical access to monitor the DUT's halted signal
        while (dut.halted == 1'b0 && cycle_counter < TIMEOUT_CYCLES) begin
            @(posedge clk);
            cycle_counter = cycle_counter + 1;
        end

        // 3. Perform self-checking
        $display("[%0t] Processor halted or timeout reached after %d cycles.", $time, cycle_counter);
        
        // Check for timeout
        if (cycle_counter >= TIMEOUT_CYCLES) begin
            $display("ERROR: Simulation timed out. Processor did not halt.");
            $display("\nSIMULATION FAILED");
            $finish;
        end

        $display("Performing final state checks...");

        // Expected values based on the pre-loaded program
        // R1 = 5
        // R2 = 10
        // R3 = R1 + R2 = 15
        // R4 = 20 (address for SW/LW)
        // data_mem[20] = R3 = 15
        // R5 = data_mem[20] = 15
        // R6 = R2 - R1 = 5
        // BEQ R1, R6 -> branch taken

        // Check Register R1
        if (dut.reg_file[1] !== 8'd5) begin
            $display("ERROR: R1 check failed. Expected: 5, Actual: %d", dut.reg_file[1]);
            $display("\nSIMULATION FAILED");
            $finish;
        end

        // Check Register R2
        if (dut.reg_file[2] !== 8'd10) begin
            $display("ERROR: R2 check failed. Expected: 10, Actual: %d", dut.reg_file[2]);
            $display("\nSIMULATION FAILED");
            $finish;
        end

        // Check Register R3
        if (dut.reg_file[3] !== 8'd15) begin
            $display("ERROR: R3 check failed. Expected: 15, Actual: %d", dut.reg_file[3]);
            $display("\nSIMULATION FAILED");
            $finish;
        end

        // Check Register R4
        if (dut.reg_file[4] !== 8'd20) begin
            $display("ERROR: R4 check failed. Expected: 20, Actual: %d", dut.reg_file[4]);
            $display("\nSIMULATION FAILED");
            $finish;
        end

        // Check Register R5 (result of LW)
        if (dut.reg_file[5] !== 8'd15) begin
            $display("ERROR: R5 check failed. Expected: 15, Actual: %d", dut.reg_file[5]);
            $display("\nSIMULATION FAILED");
            $finish;
        end

        // Check Register R6
        if (dut.reg_file[6] !== 8'd5) begin
            $display("ERROR: R6 check failed. Expected: 5, Actual: %d", dut.reg_file[6]);
            $display("\nSIMULATION FAILED");
            $finish;
        end

        // Check Data Memory (result of SW)
        if (dut.data_mem[20] !== 8'd15) begin
            $display("ERROR: Data Memory[20] check failed. Expected: 15, Actual: %d", dut.data_mem[20]);
            $display("\nSIMULATION FAILED");
            $finish;
        end

        // 4. If all checks pass, print the success message
        $display("All register and memory checks passed.");
        $display("\nSIMULATION PASSED");
        $finish;
    end

endmodule
