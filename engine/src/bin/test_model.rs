use tensorflow::{Graph, SavedModelBundle, SessionOptions, SessionRunArgs, Tensor};

fn main() {
  // In this file test_in_input is being used while in the python script,
  // that generates the saved model from Keras model it has a name "test_in".
  // For multiple inputs _input is not being appended to signature input parameter name.
  let signature_input_parameter_name = "input_1";
  let signature_output_parameter_name = "dense_1";

  // Initialize save_dir, input tensor, and an empty graph
  let save_dir = "/tmp/keras";
  let tensor: Tensor<f32> = Tensor::new(&[1024, 8, 8, 22])
    .with_values(&[1.0; 1024 * 8 * 8 * 22])
    .expect("Can't create tensor");
  let mut graph = Graph::new();

  // Load saved model bundle (session state + meta_graph data)
  let bundle = SavedModelBundle::load(&SessionOptions::new(), &["serve"], &mut graph, save_dir)
    .expect("Can't load saved model");

  // Get the session from the loaded model bundle
  let session = &bundle.session;

  // Get signature metadata from the model bundle
  let signature = bundle.meta_graph_def().get_signature("serving_default").unwrap();

  // Get input/output info
  let input_info = signature.get_input(signature_input_parameter_name).unwrap();
  let output_info = signature.get_output(signature_output_parameter_name).unwrap();

  // Get input/output ops from graph
  let input_op = graph.operation_by_name_required(&input_info.name().name).unwrap();
  let output_op = graph.operation_by_name_required(&output_info.name().name).unwrap();

  // Manages inputs and outputs for the execution of the graph
  let mut args = SessionRunArgs::new();
  args.add_feed(&input_op, 0, &tensor); // Add any inputs

  let out = args.request_fetch(&output_op, 0); // Request outputs

  // Run the model once, timing the execution
  let start = std::time::Instant::now();
  session.run(&mut args).expect("Can't run session");
  let result = args.fetch::<f32>(out).expect("Can't fetch output");
  let duration = start.elapsed();
  println!("Time elapsed in expensive_function() is: {:?}", duration);

  // Run model 100 times, timing the execution
  let start = std::time::Instant::now();
  for _ in 0..100 {
    session.run(&mut args).expect("Can't run session");
    let result = args.fetch::<f32>(out).expect("Can't fetch output");
    //println!("{:?}", result);
  }
  let duration = start.elapsed();
  println!("Time elapsed in expensive_function() is: {:?}", duration);

  // Fetch outputs after graph execution
  //let sm = args.fetch::<f32>(out).unwrap();
  //let out_res: f32 = args.fetch(out).unwrap()[0];
  //println!("Results: {:?}", sm);
}
