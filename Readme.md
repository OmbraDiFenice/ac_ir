This script attempts to extract the IR packets sent by the Mitsubishi Electric MSZ-HR35UF Air Conditioning remote.

# Requirements

 - python 3.11 or above
 - some kind of IR receiver able to dump the pulse length of the signal emitted by the remote

# Usage

Just invoke the script passing the file(s) containing the dump of the pulse length for any button press.

```sh
python3.11 ./decode.py <dump_file>
```

By default it expects Lirc [mode2](https://www.lirc.org/html/mode2.html) output in the file. Check the `--format` option for more alternatives.

See `./decode.py --help` for the full list of options.
