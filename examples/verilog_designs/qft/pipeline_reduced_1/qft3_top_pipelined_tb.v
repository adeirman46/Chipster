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

    // --- Fixed-Point Constants for Test ---
    localparam S34_ONE = 16; // 1.0 in S3.4 format (1 * 2^4)
    localparam S34_AMP = 6;  // Expected approximate amplitude of output components
    
    // --- UPDATED PIPELINE LATENCY ---
    // Total latency = 6 stages * 3 cycles/stage (H/CROT) + 1 stage * 1 cycle/stage (SWAP) = 19
    localparam PIPELINE_LATENCY = 19;

    // Clock generator
    initial begin
        clk = 0;
        forever #5 clk = ~clk; // 10ns period, 100MHz clock
    end
    
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
        i110_r = S34_ONE;

        // Wait for the pipeline to fill and for the result to be ready
        repeat(PIPELINE_LATENCY + 2) @(posedge clk);
        
        $display("Checking output after %d cycles at time %t", PIPELINE_LATENCY, $time);

        // --- Verification ---
        $display("Testing QFT on state |110> (6)");
        $display("Final State:   [ (%d,%di), (%d,%di), (%d,%di), (%d,%di), (%d,%di), (%d,%di), (%d,%di), (%d,%di) ]",
                  f000_r,f000_i, f001_r,f001_i, f010_r,f010_i, f011_r,f011_i,
                  f100_r,f100_i, f101_r,f101_i, f110_r,f110_i, f111_r,f111_i);
        $display("Expected State:  [ (6,0i), (0,-6i), (-6,0i), (0,6i), (6,0i), (0,-6i), (-6,0i), (0,6i) ] (approx.)");

        // Check against the expected S3.4 values, allowing for small rounding errors.
        // The check verifies the phase relationship (1, -i, -1, i, ...) and that the amplitude is close to the expected value.
        if (f000_r > (S34_AMP-2) && f001_i < (-S34_AMP+2) && f010_r < (-S34_AMP+2) && f011_i > (S34_AMP-2)) begin
            $display("\nResult: PASSED ✅");
        end else begin
            $display("\nResult: FAILED ❌");
        end
        
        #10 $finish;
    end

endmodule