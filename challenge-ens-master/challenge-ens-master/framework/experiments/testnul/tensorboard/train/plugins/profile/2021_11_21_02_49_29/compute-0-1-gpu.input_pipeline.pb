  *	?I?iA2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2K???^@!??j2 P@)V???@?Q@1Zm? GyB@:Preprocessing2?
iIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::ShuffleAndRepeat::ParallelMapV2::ParallelMapV2 ?ʅʿ?P@!???]?A@)?ʅʿ?P@1???]?A@:Preprocessing2?
ZIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::ShuffleAndRepeat::ParallelMapV2 ??a?J@!/-?dF3;@)??a?J@1/-?dF3;@:Preprocessing2?
KIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::ShuffleAndRepeat p???lJ@!Vϩ?;?;@)75?|???1?hpP???:Preprocessing2?
vIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::ShuffleAndRepeat::ParallelMapV2::ParallelMapV2::TensorSlice :̗`??!??_ń2??):̗`??1??_ń2??:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch3??⩯?!":D\U???)3??⩯?1":D\U???:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism??ME*???!y?ԇ'x??)DkE??ܖ?1\?A?Hׇ?:Preprocessing2F
Iterator::Model?
??捷?!?nu?ޏ??)?????1??}??p?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysisk
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.no*noZno#You may skip the rest of this page.BZ
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown
  " * 2 : B J R Z b JGPUb??No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.