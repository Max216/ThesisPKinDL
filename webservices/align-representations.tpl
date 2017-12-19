<!doctype html>

<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Align representations</title>

  <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0-beta.2/css/bootstrap.min.css" integrity="sha384-PsH8R72JQ3SOdhVi3uxftmaW6Vc51MKb0q5P2rRUpPvrszuE4W1povHYgTpBfshb" crossorigin="anonymous">

  <script src="https://code.jquery.com/jquery-3.2.1.min.js" crossorigin="anonymous"></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.12.3/umd/popper.min.js" integrity="sha384-vFJXuSJphROIrBnz7yo7oB41mKfc8JzQZiCq4NCceLEaO4IHwicKwpJf9c9IpFgh" crossorigin="anonymous"></script>
  <script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0-beta.2/js/bootstrap.min.js" integrity="sha384-alpBpkh1PFOepccYVYDB4do5UnbKysX5WZXm3XxPqe5iKTfUKjNkCk9SaVuEZflJ" crossorigin="anonymous"></script>
  <script src="/static/alignment.js"></script>
</head>

<body>
	<nav class="navbar navbar-expand-lg navbar-light bg-light">
		<p class="navbar-brand">Predict the label and see the alignments between two sentences.</p>
	</nav>

	<div style='padding:2em;'>
		<p>You can select a model, apply filters and select different features to plot. The loaded model onyl uses embeddings that have been seen in train/dev set.</p>
		<!-- Store settings -->
		<form id='form'>
			<input type="hidden" name="selected_model_index" id='selected_model_index' value='{{selected_model_idx}}'>
			% for idx, (name, path, description, acc_train, acc_dev, acc_test) in enumerate(model_options):
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
  	<div>
</body>
</html>