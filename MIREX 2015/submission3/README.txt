Music/Speech Detection Submission

Command line calling format for all executables and an example formatted set of commands:
	segment detection:
		makeMusicSpeechSegmentationMirex15(inSoundFile, outFile);

	Example:
		makeMusicSpeechSegmentationMirex15('c:\temp\xy.wav', 'c:\temp\xy.wav.seg');

Number of threads/cores used or whether this should be specified on the command line
	The algorithm is not parallel.

Expected memory footprint
	RAM: approx. 1 GB for processing a 1 hour long recording	
	
Expected runtime
	40 seconds for a 1 hour long recording

Any required environments (and versions), e.g. python, java, bash, matlab.
	Matlab 2015a, Signal processing and Statistics toolboxes
	