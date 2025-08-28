/**
 * @file risc_8bit_processor_tb.v
 * @brief Robust, self-checking testbench for the 8-bit, 5-stage pipelined RISC processor.
 *
 * @description
 * This testbench instantiates the `risc_8bit_processor` DUT, which contains a
 * hardcoded program. The testbench's role is to provide a clock and reset,
 * allow the internal program to run to completion, and then verify the final
 * state of the processor's registers and memory.
 *
 * The test sequence is as follows:
 * 1.  Generate a clock signal.
 * 2.  Apply a reset pulse to initialize the DUT to a known state.
 * 3.  Wait for the processor to execute its program and assert its internal 'halted' signal.
 * 4.  A timeout mechanism is included to prevent the simulation from running indefinitely
 *     if the 'halted' signal is never asserted.
 * 5.  Once the processor is halted, the testbench performs a series of checks on the
 *     final values of the general-purpose registers and the relevant data memory location.
 * 6.  Based on the outcome of these checks, it prints a definitive "Result: PASSED" or
 *     "Result: FAILED" message and terminates the simulation.
 *
 * @note
 * This testbench has been updated to match the optimized DUT, which replaces the
 * full data memory array with a single register (`data_mem_loc_12`) and uses
 * different values in its program (e.g., R4 is loaded with 12, not 20). The
 * checks have been adjusted accordingly.
 */
module risc_8bit_processor_tb;

    // Testbench parameters
    parameter CLK_PERIOD = 10;      // Clock period in time units
    parameter TIMEOUT_CYCLES = 200; // Max cycles to wait for the processor to halt

    // Testbench signals
    reg clk;
    reg rst;

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
        integer cycle_counter;
        integer error_found;

        // Setup for waveform dumping
        $dumpfile("risc_8bit_processor.vcd");
        $dumpvars(0, risc_8bit_processor_tb);

        $display("Starting RISC Processor Testbench...");

        // 1. Apply reset
        rst = 1'b1;
        $display("[%0t] Asserting reset.", $time);
        #(2 * CLK_PERIOD);
        rst = 1'b0;
        $display("[%0t] De-asserting reset. Processor running.", $time);

        // 2. Wait for the program to complete (or timeout)
        cycle_counter = 0;
        // Use hierarchical access to monitor the DUT's internal halted signal
        while (dut.halted == 1'b0 && cycle_counter < TIMEOUT_CYCLES) begin
            @(posedge clk);
            cycle_counter = cycle_counter + 1;
        end

        // Add a small delay for all signals to settle after the halt condition
        #10;

        $display("[%0t] Processor halted or timeout reached after %d cycles.", $time, cycle_counter);

        // 3. Check for timeout condition
        if (cycle_counter >= TIMEOUT_CYCLES) begin
            $display("ERROR: Simulation timed out. Processor did not halt.");
            $display("Result: FAILED");
            $finish;
        end

        // 4. Perform self-checking
        $display("Performing final state checks...");
        error_found = 0;

        // Expected values based on the hardcoded program execution:
        // R1 = 5
        // R2 = 10
        // R3 = R1 + R2 = 15
        // R4 = 12 (address for SW/LW)
        // data_mem_loc_12 = R3 = 15
        // R5 = data_mem_loc_12 = 15
        // R6 = R2 - R1 = 5
        // BEQ R1, R6 -> branch taken, PC jumps to 10
        // HALT

        // Check Register R1
        if (dut.reg_file[1] !== 8'd5) begin
            $display("ERROR: R1 check failed. Expected: 5, Actual: %d", dut.reg_file[1]);
            error_found = 1;
        end

        // Check Register R2
        if (dut.reg_file[2] !== 8'd10) begin
            $display("ERROR: R2 check failed. Expected: 10, Actual: %d", dut.reg_file[2]);
            error_found = 1;
        end

        // Check Register R3
        if (dut.reg_file[3] !== 8'd15) begin
            $display("ERROR: R3 check failed. Expected: 15, Actual: %d", dut.reg_file[3]);
            error_found = 1;
        end

        // Check Register R4 (Updated for the optimized DUT)
        if (dut.reg_file[4] !== 8'd12) begin
            $display("ERROR: R4 check failed. Expected: 12, Actual: %d", dut.reg_file[4]);
            error_found = 1;
        end

        // Check Register R5 (result of LW)
        if (dut.reg_file[5] !== 8'd15) begin
            $display("ERROR: R5 check failed. Expected: 15, Actual: %d", dut.reg_file[5]);
            error_found = 1;
        end

        // Check Register R6
        if (dut.reg_file[6] !== 8'd5) begin
            $display("ERROR: R6 check failed. Expected: 5, Actual: %d", dut.reg_file[6]);
            error_found = 1;
        end

        // Check Data Memory (result of SW - Updated for the optimized DUT)
        if (dut.data_mem_loc_12 !== 8'd15) begin
            $display("ERROR: Data Memory at location 12 check failed. Expected: 15, Actual: %d", dut.data_mem_loc_12);
            error_found = 1;
        end

        // 5. Print final result based on checks
        if (error_found == 0) begin
            $display("All register and memory checks passed.");
            $display("Result: PASSED");
        end else begin
            $display("One or more checks failed.");
            $display("Result: FAILED");
        end

        $finish;
    end

endmodule