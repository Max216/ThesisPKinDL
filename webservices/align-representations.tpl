<!doctype html>

<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Align representations</title>

  <link rel="stylesheet" href="/~max/bla/static/bootstrap.min.css">

  <script src="/~max/bla/static/jquery-3.2.1.min.js"></script>
  <script src="/~max/bla/static/popper.min.js"></script>
  <script src="/~max/bla/static/bootstrap.min.js"></script>
  <script src="/~max/bla/static/alignment.js"></script>
  <script type="text/javascript" src="/~max/bla/static/plotly.1.31.2.min.js"></script>
</head>

<body>
	<nav class="navbar navbar-expand-lg navbar-light bg-light">
		<p class="navbar-brand">Analyse tool.</p>
		<div class='collapse navbar-collapse' id='navbarSupportedContent'>
			<ul class='navbar-nav mr-auto'>
				<li class='nav-item active'>
					<a id='btn_word_alignment' href='#' class='nav-link'>Word alignment</a>
				</li>
				<li class='nav-item'>
					<a id='btn_general_alignment' href='#' class='nav-link'>General alignment</a>
				</li>
			</ul>
			
		</div>
	</nav>
	<div id='word-alignment-container' class='my-container'>
		<div style='padding:2em;'>
			<div class="alert alert-warning" role="alert">
			  <p>Not really made for usability. Responses might take a while.</p>
			  <p>No input checking. All required fields must have correct values.</p>
			</div>
			<p>You can select a model, apply filters and select different features to plot. The loaded model onyl uses embeddings that have been seen in train/dev set.</p>
			<!-- Store settings -->
			<form id='form'>
				<input type="hidden" name="selected_model_index" id='selected_model_index' value='{{selected_model_idx}}'>
				% for idx, (name, path, description, acc_train, acc_dev, acc_test, _) in enumerate(model_options):
					<input type="hidden" id='{{str(idx) + "-" + "description"}}' value='{{description}}'>
					<input type="hidden" id='{{str(idx) + "-" + "acc_train"}}' value='{{acc_train}}'>
					<input type="hidden" id='{{str(idx) + "-" + "acc_test"}}' value='{{acc_test}}'>
					<input type="hidden" id='{{str(idx) + "-" + "acc_dev"}}' value='{{acc_dev}}'>
				% end 

			
				<div class='container-fluid card card-body'>
					<div class='row'>
						<div class='col-sm-6'>
					  		<div id='model_selection_container'>
							  <div class="form-group">
							    <label for="form_model_selection">Select Model</label>
							    <select class='form-control' id='form_model_selection'>
							    	% for opt in model_options:
							    		<option>{{opt[0]}}</option>
						    		% end
							    </select>
							  </div>
						  	</div>
					  	</div>
					  	<div class='col-sm-6'>
					  		<p id='model_description'></p>
					  		<p> Accuracies are evaluated using batches but not sorting by premise length.</p>
					  		<p><em>Accuracy Train: </em><span id='acc_val_train'></span>%</p>
					  		<p><em>Accuracy Dev: </em><span id='acc_val_dev'></span>%</p>
					  		<p><em>Accuracy Test: </em><span id='acc_val_test'></span>%</p>
					  	</div>
				  	</div>
			  	</div>
			  	<div style="padding:1em"></div>
			  	<div class='container-fluid card card-body'>
			  		<div class='row'>
			  			<div class='col-sm-6'>
			  				<div id='feature_selection_container'>
			  					<div class='form-group'>
			  						<label for='form_feature_selection'>Select Feature for plotting</label>
			  						<select class="form-control" name='feature_selection'>
			  							<option value='nshared'>Count dimensions</option>
			  							<option value='meandiff'>Mean absolute difference</option>
			  							<option value='maxdiff'>Maximum absolute difference</option>
			  							<option value='meanc'>Mean of lower value of both</option>
			  							<option value='maxc'>Maximum of lower value of both</option>
			  							<option value='meanprod'>Mean product</option>
			  							<option value='maxprod'>Maximum product</option>
			  							<option value='minprod'>Minimum product</option>
			  						</select>
			  					</div>
			  				</div>
			  				<div style="padding:0.5em"></div>
		  					<label for='sample_category_container'>Select one of the following:</label>
		  					<div id='sample_category_container'>
	  							<div class='form-check'>
	  								<label class='form-check-label'>
	  									<input type="radio" name="shared_or_not_shared" class='form-check-input' value='shared' checked='checked'>
	  									Look at shared dimensions
	  								</label>
	  							</div>
	  							<div class='form-check'>
	  								<label class='form-check-label'>
	  									<input type="radio" name="shared_or_not_shared" class='form-check-input' value='not_shared'>
	  									Look at not shared dimensions
	  								</label>
	  							</div>
		  					</div>
			  			</div>
			  			<div class="col-sm-6">
			  				<div id='filter_container'>
			  					<p>You can filter the dimensions that are used for the plotting</p>
			  					<div class='form-check form-check-inline'>
			  						<label class='form-check-label'>
			  							<input id='filter_single_threshold_opt' type="checkbox" name="applied_filters_st" class='form-check-input' value='single_threshold_setting'>
			  							Only use dimensions where at least one value reaches this threshold
			  						</label>
			  					</div>
			  					<input id='filter_single_threshold_val' type="text" name="single_threshold_value" class='form-control col-sm-3' placeholder="e.g. 0.2">
			  					<div style="padding:0.5em"></div>
			  					<div class='form-check form-check-inline'>
			  						<label class='form-check-label'>
			  							<input id='filter_double_threshold_opt' type="checkbox" name="applied_filters_dt" class='form-check-input' value='double_threshold_setting'>
			  							Only use dimensions where BOTH values reach this threshold
			  						</label>
			  					</div>
			  					<input id='filter_double_threshold_val' type="text" name="double_threshold_value" class='form-control col-sm-3' placeholder='e.g. 0.2'>
			  				</div>
			  			</div>
			  		</div>
			  		<div id='not_shared_option_container'>
				  		<div style="padding:0.5em"></div>
				  		<div class='row'>
				  			<div class="col-sm-4">
				  				<label for='unshared_amount_meaningful_dims'>Minimum amount of meaningful dimensions</label>
				  				<input id='unshared_amount_meaningful_dims' type="text" name="unshared_amount_meaningful_dims" class='form-control col-sm-3 unshared_t' placeholder="e.g. 3">
				  			</div>
				  			<div class="col-sm-4">
				  				<label for='unshared_t_meaningful_dims'>Minimum shared threshold to consider meaningful</label>
				  				<input id='unshared_t_meaningful_dims' type="text" name="unshared_t_meaningful_dims" class='form-control col-sm-3 unshared_t' placeholder="e.g. 0.2">
				  			</div>
				  			<div class="col-sm-4">
				  				<label for='unshared_single_t'>Threshold for unshared dimansion to be considered</label>
				  				<input id='unshared_single_t unshared_t' type="text" name="unshared_single_t" class='form-control col-sm-3' placeholder="e.g. 0.2">
				  			</div>
				  		</div>
			  		</div>
			  	</div>
			  	<div style="padding:1em"></div>
			  	<div class='container-fluid card card-body'>
			  		<label for='premise_sent'>Enter premise here:</label>
			  		<input id='premise_sent' type="text" name="premise" class="form-control">
			  		<div style="padding:0.5em"></div>
			  		<label for='hyp_sent'>Enter premise here:</label>
			  		<input id='hyp_sent' type="text" name="hypothesis" class="form-control">
			  	</div>
		  		<div style="padding:1em"></div>
			  <button id='btn_run' type="button" class="btn btn-primary">Run</button>
		  	</form>

		  	<div style="padding:1em"></div>
		  	<div class='container-fluid card card-body'>
	  			<p>Prediction: <em><span id='prediction_lbl'></span></em></p>
	  			<div style="padding:0.5em"></div>
	  			<div id='result'>
	  				<figure class="figure">
					  <img id='result_image' src="" class="figure-img img-fluid">
					</figure>
	  			</div>
		  	</div>
	  	</div>
	</div>
	<div id='general-alignment-container' class='my-container'>
		<div style='padding:2em;'>
			<div class="alert alert-warning" role="alert">
			  <p>Not really made for usability. Responses might take a while.</p>
			  <p>No input checking. All required fields must have correct values.</p>
			</div>
			<div class='container-fluid card card-body'>
				<div class="row">
					<div class="col-sm-6" style="padding-right: 1.5em;">
						<p>
							Find general statistics over several sentences regarding the alignment. This is currently only possible for the model with finetuned padding. It shows statistics over 150 correctly classified samles for each label.
						</p>
						<form id='form_general'>	
							<div class='row'>
								<div class="form-group col-sm-4">
									<label for='stepsize_grid'>Bin Size</label>
									<input id='stepsize_grid' type="text" name="stepsize_grid" class='form-control' placeholder="e.g. 0.2">
								</div>
								<div class="form-check col-sm-4">
									<p> &nbsp;</p>
									<input id='cb_uncolor_center' type="checkbox" name="cb_uncolor_center" class='form-check-input activate-cb' data-activate='zero-threshold'>
									<label for='cb_uncolor_center' class='for-check-label'>Don't scale higher than this value</label> 
								</div>
								<div class="form-group col-sm-4">
									<label for='zero-threshold'>Maximum amount for coloring</label>
									<input id='zero-threshold' type="text" name="zero-threshold" class='form-control' placeholder="e.g. 30">
								</div>
							</div>
							<div class="row">
								<div class="form-group">
								    <label for="general_sample_selection">Select Type</label>
								    <select class='form-control' id='general_sample_selection' name='general_sample_selection'>
								    	% for i, opt in enumerate(general_types):
								    		<option value='{{i}}'>{{opt[0]}}</option>
							    		% end
								    </select>
							  	</div>
							</div>
						</form>
						<div class="row">
							<div class="col-sm-4">
								<button id='btn_run_general' type="button" class="btn btn-primary">Run</button>
							</div>
						</div>
					</div>
					<div class="col-sm-6" style="padding-left: 1.5em;">
						<form id='form_general_sample'>
							<p>Predict two sentences and see how many dimensions have similar values.</p>
							<div class="form-group">
								<label for="form_model_selection_general">Select Model</label>
							    <select name='model' class='form-control form_model_selection' id='form_model_selection_general'>
							    	% for i, opt in enumerate(model_options):
							    		<option value='{{i}}'>{{opt[0]}}</option>
						    		% end
							    </select>
							</div>
							<div class="form-group">
								<label for="bin_size">Bin Size</label>
								<input value="0.1" type="text" class="form-control" id="bin_size" name='bin_size' placeholder="e.g. 0.1">
							</div>
						    <div class="form-group">
								<label for="premise">Premise</label>
								<input value="A woman walking on the street." type="text" class="form-control" id="premise" name='premise' placeholder="Enter premise">
							</div>
							<div class="form-group">
								<label for="hypothesis">Hypothesis</label>
								<input value="Someone is outside." type="text" class="form-control" id="hypothesis" name='hypothesis' placeholder="Enter hypothesis">
							</div>
							<div class="form-check">
								<input id='cb_uncolor_center' type="checkbox" name="cb_uncolor_center" class='form-check-input activate-cb' data-activate='zero-threshold-sample' checked>
								<label for='cb_uncolor_center' class='for-check-label'>Don't scale higher than this value</label> 
							</div>
							<div class="form-group">
								<label for='zero-threshold-sample'>Maximum amount for coloring</label>
								<input id='zero-threshold-sample' type="text" name="zero-threshold" class='form-control' placeholder="e.g. 30" value=30>
							</div>
							<div class="row">
								<div class="col-sm-4">
									<button id='btn_run_general_sample' type="button" class="btn btn-primary">Run</button>
								</div>
							</div>
						</form>
					</div>
				</div>
			</div>
			
			
			<div style="padding:1em"></div>
		  	<div class='container-fluid card card-body hidden' id='general_averaged_results'>
	  			<figure class="figure">
	  				<figcaption class="figure-caption">Average values of correct samples (entailment).</figcaption>
  					<img src="" class="figure-img img-fluid" id='general_avg_result_entailment'>
				</figure>
				<figure class="figure">
					<figcaption class="figure-caption">Average values of correct samples (contradiction).</figcaption>
  					<img src="" class="figure-img img-fluid" id='general_avg_result_contradiction'>
				</figure>
				<figure class="figure">
					<figcaption class="figure-caption">Average values of correct samples (neutral).</figcaption>
  					<img src="" class="figure-img img-fluid" id='general_avg_result_neutral'>
				</figure>
		  	</div>
		  	<div class='container-fluid card card-body hidden' id='general_sample_results'>
		  		<p>Prediction: <span id='prediction_sample_general'></span></p>
	  			<div id="js-plot"></div>
	  			<div style="padding:1em"></div>
	  			<div>
	  				<p>You may change the representation by clicking on one of the cells.</p>
	  				<div id="selected_cell_container" class="hidden">
	  					<p>Selected: <em> start value of premise: <span id='selected_p_label'></span>, start value of hypothesis: <span id="selected_h_label"></span></em></p>
	  					<p>Move to: <em> start value of premise: <span id='selected_p_label_dest'></span>, start value of hypothesis: <span id="selected_h_label_dest"></span></em></p>
	  					<input type="hidden" id="destination_index_x">
	  					<input type="hidden" id="destination_index_y">
	  					<input type="hidden" id="origin_index_x">
	  					<input type="hidden" id="origin_index_y">
	  					<div class="row">
	  						<div class="col-sm-6">
	  							<div class="row">
	  								<div class="col-sm-4"></div>
  									<button data-dirx="0" data-diry="1" class="btn btn-primary col-sm-3 btn_direction" type="button" id='btn_up'>Up</button>
	  								<div class="col-sm-5"></div>
	  							</div>
	  							<div class="row">
	  								<div class="col-sm-1"></div>
  									<button data-dirx="-1" data-diry="0" class="btn btn-primary col-sm-3 btn_direction" type="button" id='btn_left'>Left</button>
  									<button id="btn_apply" class="btn btn-success col-sm-3" type="button">Apply</button>
  									<div id='placeholder_btn_apply' class="col-sm-3"></div>
  									<button data-dirx="1" data-diry="0" class="btn btn-primary col-sm-3 btn_direction" type="button" id='btn_right'>Right</button>
	  								<div class="col-sm-2"></div>
	  							</div>
	  							<div class="row">
	  								<div class="col-sm-4"></div>
	  								<button data-dirx="0" data-diry="-1" class="btn btn-primary col-sm-3 btn_direction" type="button" id='btn_down'>Down</button>
	  								<div class="col-sm-5"></div>
	  							</div>
	  							<div style="padding: 1em;"></div>
	  							<div class="row">
	  								<div class="col-sm-4"></div>
	  								<button id='btn_repredict' class="col-sm-3 btn btn-success">Predict</button>
	  								<div class="col-sm-5"></div>
	  							</div>
	  						</div>
	  						<div class="col-sm-6">
	  							<h3>Words:</h3>
	  							<div id="selected_cell_info"></div>
	  						</div>
	  					</div>
	  				</div>
	  			</div>
		  	</div>
	  	</div>
	</div>
</body>
</html>