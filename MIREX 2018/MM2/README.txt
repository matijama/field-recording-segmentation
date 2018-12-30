Music/Speech Detection Submission

Command line calling format for all executables and an example formatted set of commands:
Speech Detection:
		doSpeechDetection inSoundFile  outFile
Music Detection:		
		doMusicDetection inSoundFile  outFile
Music and Speech Detection:
		doMusicAndSpeechDetection inSoundFile outFile

	Example:
		doMusicAndSpeechDetection 'c:\temp\xy.wav' 'c:\temp\xy.wav.seg'

Number of threads/cores used or whether this should be specified on the command line
	Tensorflow can parallelize its execution to many cores. There is no control offered. 

Expected memory footprint
	RAM: approx. 1 GB for processing an hour long recording	
	
Expected runtime
	160 seconds for processing an hour long recording

Any required environments (and versions), e.g. python, java, bash, matlab.
	.NET Framework 4.7
	If the solution doesn't run due to not beeing able to load TfModel.dll, Visual C++ redistributable should be installed. 
        See https://visualstudio.microsoft.com/downloads/, x64 version
	