Music/Speech Classification Submission

Command line calling format for all executables and an example formatted set of commands:
	feature extracion:
		extractFeaturesMirex15(scratchPath, fileList, numberOfCores);
	training:
		trainMusicSpeechMirex15(scratchPath, fileList, numberOfCores);
	testing
		testMusicSpeechMirex15(scratchPath, fileList, outputFile, numberOfCores)

	Example:
		extractFeaturesMirex15('c:\temp','C:\mirex2015\all.txt');
		trainMusicSpeechMirex15('c:\temp','C:\mirex2015\tr1.txt');
		testMusicSpeechMirex15('c:\temp','C:\mirex2015\te1.txt','C:\mirex2015\out1.txt');


Number of threads/cores used or whether this should be specified on the command line
	Can be optionally specified on the command line, otherwise defaults to 4.

Expected memory footprint
	Disk: features take approx. 5MB per 1 hour of recordings 
	RAM: 1 Matlab thread used 200MB of memory when training on a 2.5 hour long database
	
Expected runtime
	Feature extraction takes 20 seconds for 1 hour of recordings
	1 training run takes 30 minutes on a 2.5 hour database
	Testing takes 2 seconds on a 1 hour database

Any required environments (and versions), e.g. python, java, bash, matlab.
	Matlab 2015a, Signal processing and Statistics toolboxes
	