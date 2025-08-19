`timescale 1ns/1ps
`include "fixed_point_params.vh"

module qft3_top_pipelined_tb;

    // --- Clock and Reset ---
    reg clk;
    reg rst_n;

    // --- Inputs to the DUT ---
    reg  signed [`TOTAL_WIDTH-1:0] i000_r, i000_i, i001_r, i001_i, i010_r, i010_i, i011_r, i011_i;
    reg  signed [`TOTAL_WIDTH-1:0] i100_r, i100_i, i101_r, i101_i, i110_r, i110_i, i111_r, i111_i;
    
    // --- Outputs from the DUT ---
    wire signed [`TOTAL_WIDTH-1:0] f000_r, f000_i, f001_r, f001_i, f010_r, f010_i, f011_r, f011_i;
    wire signed [`TOTAL_WIDTH-1:0] f100_r, f100_i, f101_r, f101_i, f110_r, f110_i, f111_r, f111_i;

    // --- Verification variables ---
    integer all_passed;
    reg p_0, p_1, p_2, p_3, p_4, p_5, p_6, p_7;

    // --- Instantiate the DUT ---
    qft3_top_pipelined uut (
        .clk(clk), .rst_n(rst_n),
        .i000_r(i000_r), .i000_i(i000_i), .i001_r(i001_r), .i001_i(i001_i),
        .i010_r(i010_r), .i010_i(i010_i), .i011_r(i011_r), .i011_i(i011_i),
        .i100_r(i100_r), .i100_i(i100_i), .i101_r(i101_r), .i101_i(i101_i),
        .i110_r(i110_r), .i110_i(i110_i), .i111_r(i111_r), .i111_i(i111_i),
        .f000_r(f000_r), .f000_i(f000_i), .f001_r(f001_r), .f001_i(f001_i),
        .f010_r(f010_r), .f010_i(f010_i), .f011_r(f011_r), .f011_i(f011_i),
        .f100_r(f100_r), .f100_i(f100_i), .f101_r(f101_r), .f101_i(f101_i),
        .f110_r(f110_r), .f110_i(f110_i), .f111_r(f111_r), .f111_i(f111_i)
    );

    // --- Test Parameters ---
    // Total latency = 6 stages * 3 cycles/stage (H/CROT) + 1 stage * 1 cycle/stage (SWAP) = 19
    localparam PIPELINE_LATENCY = 19;
    
    // 1.0 in S1.4 format (1 * 2^4 = 16)
    localparam S4_4_ONE = 16; 
    
    // Expected amplitude is ~ 1.0 * 1/sqrt(8) ~= 0.3535.
    // The hardware uses a constant for 1/sqrt(2) which is 11 (`11/16 = 0.6875`).
    // The final amplitude will be 1.0 * (1/sqrt(2))^3.
    // Hardware calculation: 1.0 * (11/16)^3 = 0.32495...
    // In S1.4, this is 0.32495 * 16 = 5.199... -> integer value is 5.
    localparam EXPECTED_AMP = 5;
    localparam TOLERANCE = 1;

    // Clock generator
    initial begin
        clk = 0;
        forever #5 clk = ~clk; // 10ns period, 100MHz clock
    end

    // --- Verification Task ---
    // Checks if a complex number is within tolerance of the expected value
    task check_amplitude;
        input signed [`TOTAL_WIDTH-1:0] r_val, i_val;
        input signed [`TOTAL_WIDTH-1:0] exp_r, exp_i;
        input integer tolerance;
        input [8*10:1] state_name; // Verilog-2001 style string input
        output pass; // Output declaration for the task
        reg pass;    // Internal reg for the output variable
    begin
        pass = 1;
        if (!((r_val >= (exp_r - tolerance)) && (r_val <= (exp_r + tolerance)))) begin
            $display("ERROR: Mismatch for %s [real]: Got %d, Expected %d +/- %d", state_name, r_val, exp_r, tolerance);
            pass = 0;
        end
        if (!((i_val >= (exp_i - tolerance)) && (i_val <= (exp_i + tolerance)))) begin
            $display("ERROR: Mismatch for %s [imag]: Got %d, Expected %d +/- %d", state_name, i_val, exp_i, tolerance);
            pass = 0;
        end
    end
    endtask
    
    // --- Test Sequence ---
    initial begin
        $display("--- 3-Qubit QFT Pipelined Testbench ---");
        // Initialize all inputs to zero
        {i000_r,i000_i,i001_r,i001_i,i010_r,i010_i,i011_r,i011_i} = 0;
        {i100_r,i100_i,i101_r,i101_i,i110_r,i110_i,i111_r,i111_i} = 0;

        // Pulse reset
        rst_n = 1'b0;
        #20;
        rst_n = 1'b1;
        #5;

        // Test Case: Apply QFT to the state |110> (the number 6)
        $display("Applying input state |110> (1.0) at time %t", $time);
        i110_r = S4_4_ONE;
        i110_i = 0;

        // Wait for the pipeline to fill and for the result to be ready
        repeat(PIPELINE_LATENCY + 2) @(posedge clk);
        
        $display("Checking output after %d cycles at time %t", PIPELINE_LATENCY, $time);

        // --- Verification ---
        $display("\nTesting QFT on state |110>");
        $display("Final State:      [ (%d,%di), (%d,%di), (%d,%di), (%d,%di), (%d,%di), (%d,%di), (%d,%di), (%d,%di) ]",
                 f000_r,f000_i, f001_r,f001_i, f010_r,f010_i, f011_r,f011_i,
                 f100_r,f100_i, f101_r,f101_i, f110_r,f110_i, f111_r,f111_i);
        $display("Expected State:   [ (%d,%di), (%d,%di), (%d,%di), (%d,%di), (%d,%di), (%d,%di), (%d,%di), (%d,%di) ] (approx.)",
                 EXPECTED_AMP, 0, 0, -EXPECTED_AMP, -EXPECTED_AMP, 0, 0, EXPECTED_AMP,
                 EXPECTED_AMP, 0, 0, -EXPECTED_AMP, -EXPECTED_AMP, 0, 0, EXPECTED_AMP);
        
        all_passed = 1;
        check_amplitude(f000_r, f000_i,  EXPECTED_AMP,           0, TOLERANCE, "f000", p_0); if(!p_0) all_passed = 0;
        check_amplitude(f001_r, f001_i,             0, -EXPECTED_AMP, TOLERANCE, "f001", p_1); if(!p_1) all_passed = 0;
        check_amplitude(f010_r, f010_i, -EXPECTED_AMP,           0, TOLERANCE, "f010", p_2); if(!p_2) all_passed = 0;
        check_amplitude(f011_r, f011_i,             0,  EXPECTED_AMP, TOLERANCE, "f011", p_3); if(!p_3) all_passed = 0;
        check_amplitude(f100_r, f100_i,  EXPECTED_AMP,           0, TOLERANCE, "f100", p_4); if(!p_4) all_passed = 0;
        check_amplitude(f101_r, f101_i,             0, -EXPECTED_AMP, TOLERANCE, "f101", p_5); if(!p_5) all_passed = 0;
        check_amplitude(f110_r, f110_i, -EXPECTED_AMP,           0, TOLERANCE, "f110", p_6); if(!p_6) all_passed = 0;
        check_amplitude(f111_r, f111_i,             0,  EXPECTED_AMP, TOLERANCE, "f111", p_7); if(!p_7) all_passed = 0;

        if (all_passed) begin
            $display("\nResult: PASSED ✅");
        end else begin
            $display("\nResult: FAILED ❌");
        end
        
        #10 $finish;
    end

endmodule
