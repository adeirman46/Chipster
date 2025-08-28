`include "fixed_point_params.vh"
//======================================================================
// SERIAL INTERFACE MODULE
//======================================================================
// Implements an SPI slave interface to read/write internal registers
// that connect to the parallel ports of the QFT design.
//
// SPI Protocol (Mode 0: CPOL=0, CPHA=0):
// - Transaction begins when `cs` (active low) goes low.
// - Master sends an 8-bit command byte.
// - Master then sends/receives an 8-bit data byte.
// - Transaction ends when `cs` goes high.
//
// Command Byte Format:
// - Bit 7: R/W (1 for Read, 0 for Write)
// - Bit 6: Unused
// - Bits 5-0: Address (6 bits, 0-63)
//
// Data Byte Format:
// - Bits 7-6: Unused (ignored on write, zero on read)
// - Bits 5-0: Data (`TOTAL_WIDTH`-bit value)
//
// Address Map:
// - 0x00 - 0x0F: Write-only registers for QFT inputs (i000_r to i111_i)
// - 0x10 - 0x1F: Read-only registers for QFT outputs (f000_r to f111_i)
//
module spi_interface(
    // System Interface
    input clk,
    input rst_n,

    // SPI Interface
    input sclk,
    input cs,
    input mosi,
    output miso,

    // Parallel Interface to QFT inputs (16 total)
    output reg signed [`TOTAL_WIDTH-1:0] out_i000_r, output reg signed [`TOTAL_WIDTH-1:0] out_i000_i,
    output reg signed [`TOTAL_WIDTH-1:0] out_i001_r, output reg signed [`TOTAL_WIDTH-1:0] out_i001_i,
    output reg signed [`TOTAL_WIDTH-1:0] out_i010_r, output reg signed [`TOTAL_WIDTH-1:0] out_i010_i,
    output reg signed [`TOTAL_WIDTH-1:0] out_i011_r, output reg signed [`TOTAL_WIDTH-1:0] out_i011_i,
    output reg signed [`TOTAL_WIDTH-1:0] out_i100_r, output reg signed [`TOTAL_WIDTH-1:0] out_i100_i,
    output reg signed [`TOTAL_WIDTH-1:0] out_i101_r, output reg signed [`TOTAL_WIDTH-1:0] out_i101_i,
    output reg signed [`TOTAL_WIDTH-1:0] out_i110_r, output reg signed [`TOTAL_WIDTH-1:0] out_i110_i,
    output reg signed [`TOTAL_WIDTH-1:0] out_i111_r, output reg signed [`TOTAL_WIDTH-1:0] out_i111_i,

    // Parallel Interface from QFT outputs (16 total)
    input signed [`TOTAL_WIDTH-1:0] in_f000_r, input signed [`TOTAL_WIDTH-1:0] in_f000_i,
    input signed [`TOTAL_WIDTH-1:0] in_f001_r, input signed [`TOTAL_WIDTH-1:0] in_f001_i,
    input signed [`TOTAL_WIDTH-1:0] in_f010_r, input signed [`TOTAL_WIDTH-1:0] in_f010_i,
    input signed [`TOTAL_WIDTH-1:0] in_f011_r, input signed [`TOTAL_WIDTH-1:0] in_f011_i,
    input signed [`TOTAL_WIDTH-1:0] in_f100_r, input signed [`TOTAL_WIDTH-1:0] in_f100_i,
    input signed [`TOTAL_WIDTH-1:0] in_f101_r, input signed [`TOTAL_WIDTH-1:0] in_f101_i,
    input signed [`TOTAL_WIDTH-1:0] in_f110_r, input signed [`TOTAL_WIDTH-1:0] in_f110_i,
    input signed [`TOTAL_WIDTH-1:0] in_f111_r, input signed [`TOTAL_WIDTH-1:0] in_f111_i
);

    // FSM states
    localparam S_IDLE = 3'd0;
    localparam S_CMD  = 3'd1;
    localparam S_WAIT = 3'd2;
    localparam S_READ = 3'd3;
    localparam S_WRITE= 3'd4;
    localparam S_EXEC = 3'd5;

    reg [2:0] state;

    // SPI signal synchronization and edge detection
    reg sclk_d1, sclk_d2;
    wire sclk_posedge;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sclk_d1 <= 1'b0;
            sclk_d2 <= 1'b0;
        end else begin
            sclk_d1 <= sclk;
            sclk_d2 <= sclk_d1;
        end
    end
    assign sclk_posedge = (sclk_d1 == 1'b1) && (sclk_d2 == 1'b0);

    // Internal registers
    reg [3:0] bit_cnt;
    reg [7:0] mosi_sreg;
    reg [7:0] miso_sreg;
    reg [7:0] cmd_reg;
    
    wire is_read = cmd_reg[7];
    wire [5:0] addr = cmd_reg[5:0];

    // MISO output logic
    assign miso = cs ? 1'bz : miso_sreg[7];

    // FSM and Data Logic
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
            bit_cnt <= 4'd0;
            mosi_sreg <= 8'd0;
            miso_sreg <= 8'd0;
            cmd_reg <= 8'd0;
        end else begin
            // Abort transaction if cs goes high
            if (cs) begin
                state <= S_IDLE;
                bit_cnt <= 4'd0;
            end else begin
                case (state)
                    S_IDLE: begin
                        if (!cs) begin
                            state <= S_CMD;
                            bit_cnt <= 4'd0;
                        end
                    end
                    S_CMD: begin
                        if (sclk_posedge) begin
                            mosi_sreg <= {mosi_sreg[6:0], mosi};
                            bit_cnt <= bit_cnt + 1;
                            if (bit_cnt == 4'd7) begin
                                cmd_reg <= {mosi_sreg[6:0], mosi};
                                state <= S_WAIT;
                            end
                        end
                    end
                    S_WAIT: begin
                        bit_cnt <= 4'd0;
                        if (is_read) begin
                            state <= S_READ;
                            // Pre-load miso shift register based on read address
                            case (addr)
                                6'h10: miso_sreg <= {2'b0, in_f000_r};
                                6'h11: miso_sreg <= {2'b0, in_f000_i};
                                6'h12: miso_sreg <= {2'b0, in_f001_r};
                                6'h13: miso_sreg <= {2'b0, in_f001_i};
                                6'h14: miso_sreg <= {2'b0, in_f010_r};
                                6'h15: miso_sreg <= {2'b0, in_f010_i};
                                6'h16: miso_sreg <= {2'b0, in_f011_r};
                                6'h17: miso_sreg <= {2'b0, in_f011_i};
                                6'h18: miso_sreg <= {2'b0, in_f100_r};
                                6'h19: miso_sreg <= {2'b0, in_f100_i};
                                6'h1A: miso_sreg <= {2'b0, in_f101_r};
                                6'h1B: miso_sreg <= {2'b0, in_f101_i};
                                6'h1C: miso_sreg <= {2'b0, in_f110_r};
                                6'h1D: miso_sreg <= {2'b0, in_f110_i};
                                6'h1E: miso_sreg <= {2'b0, in_f111_r};
                                6'h1F: miso_sreg <= {2'b0, in_f111_i};
                                default: miso_sreg <= 8'h00;
                            endcase
                        end else begin
                            state <= S_WRITE;
                        end
                    end
                    S_READ: begin
                        if (sclk_posedge) begin
                            miso_sreg <= {miso_sreg[6:0], 1'b0}; // Shift out data
                            bit_cnt <= bit_cnt + 1;
                            if (bit_cnt == 4'd7) begin
                                state <= S_IDLE; // End of transaction
                            end
                        end
                    end
                    S_WRITE: begin
                        if (sclk_posedge) begin
                            mosi_sreg <= {mosi_sreg[6:0], mosi};
                            bit_cnt <= bit_cnt + 1;
                            if (bit_cnt == 4'd7) begin
                                state <= S_EXEC;
                            end
                        end
                    end
                    S_EXEC: begin
                        // This state lasts one clock cycle to execute the write
                        state <= S_IDLE;
                    end
                    default: state <= S_IDLE;
                endcase
            end
        end
    end

    // Parallel register write logic
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_i000_r <= 0; out_i000_i <= 0; out_i001_r <= 0; out_i001_i <= 0;
            out_i010_r <= 0; out_i010_i <= 0; out_i011_r <= 0; out_i011_i <= 0;
            out_i100_r <= 0; out_i100_i <= 0; out_i101_r <= 0; out_i101_i <= 0;
            out_i110_r <= 0; out_i110_i <= 0; out_i111_r <= 0; out_i111_i <= 0;
        end else if (state == S_EXEC) begin
            case (addr)
                6'h00: out_i000_r <= mosi_sreg[`TOTAL_WIDTH-1:0];
                6'h01: out_i000_i <= mosi_sreg[`TOTAL_WIDTH-1:0];
                6'h02: out_i001_r <= mosi_sreg[`TOTAL_WIDTH-1:0];
                6'h03: out_i001_i <= mosi_sreg[`TOTAL_WIDTH-1:0];
                6'h04: out_i010_r <= mosi_sreg[`TOTAL_WIDTH-1:0];
                6'h05: out_i010_i <= mosi_sreg[`TOTAL_WIDTH-1:0];
                6'h06: out_i011_r <= mosi_sreg[`TOTAL_WIDTH-1:0];
                6'h07: out_i011_i <= mosi_sreg[`TOTAL_WIDTH-1:0];
                6'h08: out_i100_r <= mosi_sreg[`TOTAL_WIDTH-1:0];
                6'h09: out_i100_i <= mosi_sreg[`TOTAL_WIDTH-1:0];
                6'h0A: out_i101_r <= mosi_sreg[`TOTAL_WIDTH-1:0];
                6'h0B: out_i101_i <= mosi_sreg[`TOTAL_WIDTH-1:0];
                6'h0C: out_i110_r <= mosi_sreg[`TOTAL_WIDTH-1:0];
                6'h0D: out_i110_i <= mosi_sreg[`TOTAL_WIDTH-1:0];
                6'h0E: out_i111_r <= mosi_sreg[`TOTAL_WIDTH-1:0];
                6'h0F: out_i111_i <= mosi_sreg[`TOTAL_WIDTH-1:0];
            endcase
        end
    end

endmodule
