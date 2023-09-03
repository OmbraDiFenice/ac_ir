import math
import argparse
import abc
from dataclasses import dataclass
from textwrap import wrap


# transmission:
# prologue - packet - wait - prologue - packet - wait
#
# where:
#  prologue: 8*DUTY_CYCLE pulse - 4*DUTY_CYCLE space
#  packet:
#    logic 0: 1*DUTY_CYCLE pulse - 1*DUTY_CYCLE space
#    logic 1: 1*DUTY_CYCLE pulse - 3*DUTY_CYCLE space
#  wait: > 20*DUTY_CYCLE space


DUTY_CYCLE = 330
FIRST = 'first'
SECOND = 'second'


@dataclass
class DecodedSignal:
    name: str
    str_data: list[str]

    @property
    def data(self) -> list[bytearray]:
        bin_packets = []
        for str_packet in self.str_data:
            binary = []
            byte = 0
            bit_len = 0
            for bit in str_packet:
                byte = (byte << 1) | (1 if bit == '1' else 0)
                bit_len += 1
                if bit_len == 8:
                    binary.append(byte)
                    byte = 0
                    bit_len = 0
            if bit_len < 8:
                while bit_len < 8:
                    byte = (byte << 1)
                    bit_len += 1
                binary.append(byte)
            bin_packets.append(binary)

        return [bytearray(b) for b in bin_packets if len(b) > 1]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('filenames', nargs='*', help='list of files to decode. They should contain signal data according to the --format option')

    input_options = parser.add_argument_group('input decoding options')
    input_options.add_argument('-t', '--timeout-check', action='store_true', help='validate that each file contains only 1 timeout. Can be enabled when the file is expected to contain codes only for 1 keypress')
    input_options.add_argument('-f', '--format', choices=['lirc', 'flipper'], default='lirc', help='input file format. If lirc the input files are expected to contain the dump from `mode2` program')

    display_options = parser.add_argument_group('Display option')
    display_options.add_argument('-s', '--signal-name', help='limit the output to the signal with the given name. If --format=flipper the name comes from the flipper dump. If --format=lirc the name is the file name containing the mode2 dump. If not specified display all input signals')
    display_options.add_argument('-b', '--binary', action='store_true', help='display the decoded message in binary, otherwise in hex. Note that in hex format the data is padded with 0s at the end if the number of bits of the packet is not multiple of 8')
    display_options.add_argument('-d', '--doubles', action='store_true', help='display duplicate consecutive packets, hide them otherwise')

    postprocessor_options = parser.add_argument_group('Post-processor options', 'These operations are applied to each packet in each signal')
    postprocessor_options.add_argument('-E', '--big-endian', action='store_true', help='interpret and show data as big endian bytes')
    postprocessor_options.add_argument('-H', '--header', type=int, default=8*8, help='length in bits of a constant header found at the beginning of each packet. Those bits are ignored and not displayed in the output. NOTE: this is applied before --remove-double-bit')
    postprocessor_options.add_argument('-C', '--checksum-bits', type=int, default=18, help='number of bits of the checksum in the signal. The checksum is assumed to be at the end of the signal. --remove-double-bit will ignore the checksum bits')
    postprocessor_options.add_argument('--remove-double-bit', choices=[FIRST, SECOND], help='only pick the first or second bit in any consecutive pair of 2 bits from the signal. This reduces the length of the signal by a factor of 2. It doesn\'t apply to the checksum bytes if specified, see --checksum-start-byte')
    postprocessor_options.add_argument('--raw', action='store_true', help='skip any post-processing and display the raw data as decoded from the input file')

    return parser.parse_args()


class Decoder(abc.ABC):
    def decode(self, filename: str, options: argparse.Namespace) -> list[DecodedSignal]:
        if not self._validate(filename, options):
            print(f'{filename} invalid')
            return []

        signals = self._extract_signals(filename)

        decoded_signals = []
        for signal_name, signal_data in signals.items():
            bit_strings = self._decode_to_bit_string(signal_data, options)

            decoded_signals.append(DecodedSignal(
                name=signal_name,
                str_data=bit_strings,
            ))

        return decoded_signals

    @abc.abstractmethod
    def _extract_signals(self, filename: str) -> dict[str, list[int]]:
        pass

    @abc.abstractmethod
    def _validate(self, filename: str, options: argparse.Namespace) -> bool:
        pass

    def _decode_to_bit_string(self, raw_data: list[int], options: argparse.Namespace) -> list[str]:
        packets = []
        packet = ''

        i = 0
        while i < len(raw_data) - 1:
            first = raw_data[i]

            if self._is_wait(first):
                packets.append(packet)
                i += 1
                continue

            i += 1
            second = raw_data[i]

            if self._is_prologue(first, second):
                packets.append(packet)
                packet = ''
            else:
                packet += self._decode_bit(first, second)

        return packets

    def _normalize(self, pulse: int, space: int) -> (int, int):
        normalized_pulse = math.floor(pulse / DUTY_CYCLE)
        normalized_space = math.floor(space / DUTY_CYCLE)

        return normalized_pulse, normalized_space

    def _is_prologue(self, pulse: int, space: int) -> bool:
        normalized_pulse, normalized_space = self._normalize(pulse, space)
        return normalized_pulse >= 3 and normalized_space >= 3

    def _is_wait(self, signal: int) -> bool:
        return signal > 20 * DUTY_CYCLE

    def _decode_bit(self, pulse: int, space: int) -> str:
        normalized_pulse, normalized_space = self._normalize(pulse, space)
        return '0' if abs(normalized_pulse - normalized_space) < 2 else '1'


class LircDecoder(Decoder):
    def _validate(self, filename: str, options: argparse.Namespace) -> bool:
        found_pulse = False
        found_space = False
        found_timeout = False

        with open(filename) as f:
            for line in f:
                if options.timeout_check and found_timeout:
                    print('timeout should be the last line')
                    return False

                signal = line.split()

                if signal[0] == 'pulse':
                    if found_pulse:
                        print('invalid sequence: 2 pulse in a row')
                        return False
                    found_pulse = True

                if signal[0] == 'space':
                    if not found_pulse or found_space:
                        print('invalid sequence: 2 space in a row')
                        return False
                    found_space = True

                if found_pulse and found_space:
                    found_pulse = False
                    found_space = False

                if signal[0] == 'timeout':
                    found_timeout = True
                    if not options.timeout_check:
                        found_pulse = False
                        found_space = False

        return True

    def _extract_signals(self, filename: str) -> dict[str, list[int]]:
        with open(filename) as f:
            return {filename: [int(line.split()[1]) for line in f]}


class FlipperDecoder(Decoder):
    def _validate(self, filename: str, options: argparse.Namespace) -> bool:
        return True

    def _extract_signals(self, filename: str) -> dict[str, list[int]]:
        signals = {}

        with open(filename) as f:
            line = f.readline()
            while line != '':
                while not line.strip().startswith('#'):
                    line = f.readline()

                signal_name = f.readline().split(':')[1].strip()
                signal_type = f.readline().split(':')[1].strip()
                signal_frequency = int(f.readline().split(':')[1].strip())
                signal_duty_cycle = float(f.readline().split(':')[1].strip())
                signal_data = [int(x) for x in f.readline().split(':')[1].strip().split()]

                signals[signal_name] = signal_data

                line = f.readline()

        return signals


class PostProcessor:
    def process(self, signal: DecodedSignal, options: argparse.Namespace) -> DecodedSignal:
        return DecodedSignal(
            name=signal.name,
            str_data=[self._process_packet(packet, options) for packet in signal.str_data]
        )

    def _process_packet(self, packet: str, options: argparse.Namespace) -> str:
        packet = self._skip_constant_prefix(packet, options.header)

        if options.big_endian:
            packet = self._to_big_endian(packet)

        packet, checksum = self._extract_checksum(packet, options.checksum_bits)

        if options.remove_double_bit:
            packet = self._remove_double_bits(packet, options.remove_double_bit)

        #packet = self._rotate_pairs_of_bytes(packet)

        packet = packet + checksum
        return packet

    def _extract_checksum(self, packet: str, checksum_bit_len: int) -> tuple[str, str]:
        """Returns a tuple where the first element is the packet data and the second the checksum bits"""
        return packet[:len(packet)-checksum_bit_len], packet[len(packet)-checksum_bit_len:]

    def _skip_constant_prefix(self, packet: str, bits_to_skip: int) -> str:
        return packet[bits_to_skip:]

    def _rotate_pairs_of_bytes(self, packet: str) -> str:
        return ''.join(b[1:]+b[0] for b in wrap(packet, 8))

    def _remove_double_bits(self, packet: str, bit_to_keep: str) -> str:
        def keep(index: int) -> bool:
            return (index % 2 == 0) if bit_to_keep == SECOND else (index % 2 == 1)
        return ''.join(bit for i, bit in enumerate(packet) if keep(i))

    def _to_big_endian(self, packet: str) -> str:
        big_endian_packet = ''
        for byte in wrap(packet, 8):
            big_endian_byte = byte[::-1]
            big_endian_packet += big_endian_byte
        return big_endian_packet


def to_raw_bin(bit_string: str) -> str:
    return ' '.join([
        b for b in wrap(bit_string, 8)
    ])


def print_decoded(decoded: DecodedSignal, options: argparse.Namespace) -> None:
    packets = decoded.data

    if options.binary:
        packets = [to_raw_bin(b) for b in decoded.str_data]
    else:
        packets = [d.hex(' ') for d in packets]

    last_packet = ''
    print(decoded.name)
    print('packet  data')
    for i, packet in enumerate(packets):
        if options.doubles or packet != last_packet:
            print(f'{i+1:02}/{len(packets):02}   {packet}')
            last_packet = packet


def main():
    options = parse_arguments()
    post_processor = PostProcessor()
    decoder = LircDecoder() if options.format == 'lirc' else FlipperDecoder()

    for filename in options.filenames:
        decoded = decoder.decode(filename, options)
        for decoded_signal in decoded:
            if options.signal_name is not None and options.signal_name != decoded_signal.name:
                continue
            if not options.raw:
                processed_signal = post_processor.process(decoded_signal, options)
                print_decoded(processed_signal, options)
            else:
                print_decoded(decoded_signal, options)
            print()


if __name__ == '__main__':
    main()
