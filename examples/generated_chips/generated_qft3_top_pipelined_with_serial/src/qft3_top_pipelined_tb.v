`timescale 1ns/1ps
`include "fixed_point_params.vh"

module qft3_top_pipelined_with_serial_tb;

    // --- Testbench Parameters ---
    localparam CLK_PERIOD = 10;
    localparam SCLK_PERIOD = 40; // SCLK must be slower than CLK for the synchronizer
    localparam SCLK_HALF_PERIOD = SCLK_PERIOD / 2;

    // --- DUT Interface ---
    reg clk;
    reg rst_n;
    reg sclk;
    reg cs;
    reg mosi;
    wire miso;

    // --- Instantiate the DUT ---
    qft3_top_pipelined_with_serial uut (
        .clk(clk),
        .rst_n(rst_n),
        .sclk(sclk),
        .cs(cs),
        .mosi(mosi),
        .miso(miso)
    );

    // --- Test Parameters from original TB ---
    // Total latency = 6 stages * 3 cycles/stage (H/CROT) + 1 stage * 1 cycle/stage (SWAP) = 19
    localparam PIPELINE_LATENCY = 19;
    
    // 1.0 in S1.4 format (1 * 2^4 = 16)
    localparam S4_4_ONE = 16; 
    
    // Expected amplitude is ~ 1.0 * 1/sqrt(8) ~= 0.3535.
    // Hardware calculation: 1.0 * (11/16)^3 = 0.32495...
    // In S1.4, this is 0.32495 * 16 = 5.199... -> integer value is 5.
    localparam EXPECTED_AMP = 5;
    localparam TOLERANCE = 1;

    // --- Verification variables ---
    integer all_passed;
    reg p_0, p_1, p_2, p_3, p_4, p_5, p_6, p_7;
    // Arrays to hold results read back from DUT
    reg signed [`TOTAL_WIDTH-1:0] tb_f_r [0:7];
    reg signed [`TOTAL_WIDTH-1:0] tb_f_i [0:7];

    // --- Clock generator ---
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end

    //======================================================================
    // SPI MASTER TASKS
    //======================================================================

    // Task to perform a single SPI write transaction (16 bits)
    task spi_write;
        input [5:0] addr;
        input [`TOTAL_WIDTH-1:0] data;
        integer i;
        reg [7:0] cmd_byte;
        reg [7:0] data_byte;

    begin
        cmd_byte = {1'b0, 1'b0, addr}; // Write command
        data_byte = {2'b00, data};

        // Start transaction
        cs = 1'b0;
        sclk = 1'b0;
        #(SCLK_HALF_PERIOD/2);

        // Shift out command byte (MSB first)
        for (i = 7; i >= 0; i = i - 1) begin
            mosi = cmd_byte[i];
            #SCLK_HALF_PERIOD;
            sclk = 1'b1;
            #SCLK_HALF_PERIOD;
            sclk = 1'b0;
        end
        
        // Shift out data byte (MSB first)
        for (i = 7; i >= 0; i = i - 1) begin
            mosi = data_byte[i];
            #SCLK_HALF_PERIOD;
            sclk = 1'b1;
            #SCLK_HALF_PERIOD;
            sclk = 1'b0;
        end

        #(SCLK_HALF_PERIOD/2);
        // End transaction
        cs = 1'b1;
        mosi = 1'bz; // Tristate MOSI when not in use
        @(posedge clk); // Wait one main clock cycle between transactions
    end
    endtask

    // Task to perform a single SPI read transaction (16 bits)
    task spi_read;
        input [5:0] addr;
        output [`TOTAL_WIDTH-1:0] data;
        integer i;
        reg [7:0] cmd_byte;
        reg [7:0] received_byte;

    begin
        cmd_byte = {1'b1, 1'b0, addr}; // Read command

        // Start transaction
        cs = 1'b0;
        sclk = 1'b0;
        #(SCLK_HALF_PERIOD/2);

        // Shift out command byte (MSB first)
        for (i = 7; i >= 0; i = i - 1) begin
            mosi = cmd_byte[i];
            #SCLK_HALF_PERIOD;
            sclk = 1'b1;
            #SCLK_HALF_PERIOD;
            sclk = 1'b0;
        end

        mosi = 1'bz; // Let slave drive MISO
        
        // Shift in data byte (MSB first)
        for (i = 7; i >= 0; i = i - 1) begin
            #SCLK_HALF_PERIOD;
            sclk = 1'b1;
            received_byte[i] = miso;
            #SCLK_HALF_PERIOD;
            sclk = 1'b0;
        end

        #(SCLK_HALF_PERIOD/2);
        // End transaction
        cs = 1'b1;
        @(posedge clk); // Wait one main clock cycle between transactions

        data = received_byte[`TOTAL_WIDTH-1:0];
    end
    endtask

    // --- Verification Task (copied from original TB) ---
    task check_amplitude;
        input signed [`TOTAL_WIDTH-1:0] r_val, i_val;
        input signed [`TOTAL_WIDTH-1:0] exp_r, exp_i;
        input integer tolerance;
        input [8*10:1] state_name;
        output pass;
        reg pass;
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

    // --- Main Test Sequence ---
    initial begin
        integer i;
        $display("--- 3-Qubit QFT Pipelined with Serial Interface Testbench ---");
        
        // Initialize SPI master signals
        sclk = 1'b0;
        cs = 1'b1;
        mosi = 1'bz;

        // Pulse reset
        rst_n = 1'b0;
        #20;
        rst_n = 1'b1;
        @(posedge clk);

        // --- PHASE 1: Write input vector via SPI ---
        // Test Case: Apply QFT to the state |110> (the number 6)
        $display("Applying input state |110> (1.0) via SPI at time %t", $time);

        // Write 0 to all 16 input registers first
        for (i = 0; i < 16; i = i + 1) begin
            spi_write(i, 0);
        end

        // Write 1.0 to i110_r (address 0x0C)
        spi_write(6'h0C, S4_4_ONE);
        $display("Finished writing inputs at time %t", $time);

        // --- PHASE 2: Wait for DUT processing ---
        // Wait for the pipeline to fill and for the result to be ready
        $display("Waiting %d cycles for pipeline to complete...", PIPELINE_LATENCY);
        repeat(PIPELINE_LATENCY + 2) @(posedge clk);
        
        // --- PHASE 3: Read output vector via SPI ---
        $display("Reading output vector via SPI at time %t", $time);
        for (i = 0; i < 8; i = i + 1) begin
            spi_read(6'h10 + (2*i),   tb_f_r[i]);
            spi_read(6'h10 + (2*i)+1, tb_f_i[i]);
        end
        $display("Finished reading outputs at time %t", $time);

        // --- PHASE 4: Verification ---
        $display("\nTesting QFT on state |110>");
        $display("Final State:      [ (%d,%di), (%d,%di), (%d,%di), (%d,%di), (%d,%di), (%d,%di), (%d,%di), (%d,%di) ]",
                 tb_f_r[0],tb_f_i[0], tb_f_r[1],tb_f_i[1], tb_f_r[2],tb_f_i[2], tb_f_r[3],tb_f_i[3],
                 tb_f_r[4],tb_f_i[4], tb_f_r[5],tb_f_i[5], tb_f_r[6],tb_f_i[6], tb_f_r[7],tb_f_i[7]);
        $display("Expected State:   [ (%d,%di), (%d,%di), (%d,%di), (%d,%di), (%d,%di), (%d,%di), (%d,%di), (%d,%di) ] (approx.)",
                 EXPECTED_AMP, 0, 0, -EXPECTED_AMP, -EXPECTED_AMP, 0, 0, EXPECTED_AMP,
                 EXPECTED_AMP, 0, 0, -EXPECTED_AMP, -EXPECTED_AMP, 0, 0, EXPECTED_AMP);
        
        all_passed = 1;
        check_amplitude(tb_f_r[0], tb_f_i[0],  EXPECTED_AMP,           0, TOLERANCE, "f000", p_0); if(!p_0) all_passed = 0;
        check_amplitude(tb_f_r[1], tb_f_i[1],             0, -EXPECTED_AMP, TOLERANCE, "f001", p_1); if(!p_1) all_passed = 0;
        check_amplitude(tb_f_r[2], tb_f_i[2], -EXPECTED_AMP,           0, TOLERANCE, "f010", p_2); if(!p_2) all_passed = 0;
        check_amplitude(tb_f_r[3], tb_f_i[3],             0,  EXPECTED_AMP, TOLERANCE, "f011", p_3); if(!p_3) all_passed = 0;
        check_amplitude(tb_f_r[4], tb_f_i[4],  EXPECTED_AMP,           0, TOLERANCE, "f100", p_4); if(!p_4) all_passed = 0;
        check_amplitude(tb_f_r[5], tb_f_i[5],             0, -EXPECTED_AMP, TOLERANCE, "f101", p_5); if(!p_5) all_passed = 0;
        check_amplitude(tb_f_r[6], tb_f_i[6], -EXPECTED_AMP,           0, TOLERANCE, "f110", p_6); if(!p_6) all_passed = 0;
        check_amplitude(tb_f_r[7], tb_f_i[7],             0,  EXPECTED_AMP, TOLERANCE, "f111", p_7); if(!p_7) all_passed = 0;

        if (all_passed) begin
            $display("\nResult: PASSED");
        end else begin
            $display("\nResult: FAILED");
        end
        
        #10 $finish;
    end

endmodule