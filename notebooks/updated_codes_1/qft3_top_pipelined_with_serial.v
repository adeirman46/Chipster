`include "fixed_point_params.vh"
//======================================================================
// NEW TOP-LEVEL MODULE
//======================================================================
// This module instantiates the original QFT design and the new SPI
// interface, connecting them to create a pin-reduced system.
// Total Pin Count: 6 (clk, rst_n, sclk, cs, mosi, miso)
//
module qft3_top_pipelined_with_serial(
    // System I/O
    input clk,
    input rst_n,

    // Serial Interface I/O
    input sclk,
    input cs,
    input mosi,
    output miso
);

    // Wires to connect SPI interface to QFT core
    // --- QFT Inputs ---
    wire signed [`TOTAL_WIDTH-1:0] w_i000_r, w_i000_i, w_i001_r, w_i001_i;
    wire signed [`TOTAL_WIDTH-1:0] w_i010_r, w_i010_i, w_i011_r, w_i011_i;
    wire signed [`TOTAL_WIDTH-1:0] w_i100_r, w_i100_i, w_i101_r, w_i101_i;
    wire signed [`TOTAL_WIDTH-1:0] w_i110_r, w_i110_i, w_i111_r, w_i111_i;

    // --- QFT Outputs ---
    wire signed [`TOTAL_WIDTH-1:0] w_f000_r, w_f000_i, w_f001_r, w_f001_i;
    wire signed [`TOTAL_WIDTH-1:0] w_f010_r, w_f010_i, w_f011_r, w_f011_i;
    wire signed [`TOTAL_WIDTH-1:0] w_f100_r, w_f100_i, w_f101_r, w_f101_i;
    wire signed [`TOTAL_WIDTH-1:0] w_f110_r, w_f110_i, w_f111_r, w_f111_i;

    // Instantiate the SPI interface module
    spi_interface spi_inst (
        .clk(clk),
        .rst_n(rst_n),
        .sclk(sclk),
        .cs(cs),
        .mosi(mosi),
        .miso(miso),
        // Connect SPI outputs to QFT input wires
        .out_i000_r(w_i000_r), .out_i000_i(w_i000_i), .out_i001_r(w_i001_r), .out_i001_i(w_i001_i),
        .out_i010_r(w_i010_r), .out_i010_i(w_i010_i), .out_i011_r(w_i011_r), .out_i011_i(w_i011_i),
        .out_i100_r(w_i100_r), .out_i100_i(w_i100_i), .out_i101_r(w_i101_r), .out_i101_i(w_i101_i),
        .out_i110_r(w_i110_r), .out_i110_i(w_i110_i), .out_i111_r(w_i111_r), .out_i111_i(w_i111_i),
        // Connect QFT output wires to SPI inputs
        .in_f000_r(w_f000_r), .in_f000_i(w_f000_i), .in_f001_r(w_f001_r), .in_f001_i(w_f001_i),
        .in_f010_r(w_f010_r), .in_f010_i(w_f010_i), .in_f011_r(w_f011_r), .in_f011_i(w_f011_i),
        .in_f100_r(w_f100_r), .in_f100_i(w_f100_i), .in_f101_r(w_f101_r), .in_f101_i(w_f101_i),
        .in_f110_r(w_f110_r), .in_f110_i(w_f110_i), .in_f111_r(w_f111_r), .in_f111_i(w_f111_i)
    );

    // Instantiate the original QFT core module
    qft3_top_pipelined qft_core_inst (
        .clk(clk),
        .rst_n(rst_n),
        // Connect inputs from SPI wires
        .i000_r(w_i000_r), .i000_i(w_i000_i), .i001_r(w_i001_r), .i001_i(w_i001_i),
        .i010_r(w_i010_r), .i010_i(w_i010_i), .i011_r(w_i011_r), .i011_i(w_i011_i),
        .i100_r(w_i100_r), .i100_i(w_i100_i), .i101_r(w_i101_r), .i101_i(w_i101_i),
        .i110_r(w_i110_r), .i110_i(w_i110_i), .i111_r(w_i111_r), .i111_i(w_i111_i),
        // Connect outputs to SPI wires
        .f000_r(w_f000_r), .f000_i(w_f000_i), .f001_r(w_f001_r), .f001_i(w_f001_i),
        .f010_r(w_f010_r), .f010_i(w_f010_i), .f011_r(w_f011_r), .f011_i(w_f011_i),
        .f100_r(w_f100_r), .f100_i(w_f100_i), .f101_r(w_f101_r), .f101_i(w_f101_i),
        .f110_r(w_f110_r), .f110_i(w_f110_i), .f111_r(w_f111_r), .f111_i(w_f111_i)
    );

endmodule
