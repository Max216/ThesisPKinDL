
function updateUnsharedOptions() {
	active = $('input[name=shared_or_not_shared]:checked').val()
	if (active == 'shared') {
		$('#not_shared_option_container').hide()
	} else{
		$('#not_shared_option_container').show()
	}
}

function updateFilterFields(){
	filter_single_threshold = $('#filter_single_threshold_opt:checked').length > 0
	filter_double_threshold = $('#filter_double_threshold_opt:checked').length > 0
	
	$('#filter_single_threshold_val').prop('disabled', !filter_single_threshold)
	$('#filter_double_threshold_val').prop('disabled', !filter_double_threshold)

}

function updateModelInfo() {
	index = $('#selected_model_index').val()
	$('#form_model_selection option').eq(index).prop('selected', true);
	$('#selected_model_index').val(index)

	modelDescription = $('#' + index + '-description').val()
	modelAccTrain = $('#' + index + '-acc_train').val()
	modelAccDev = $('#' + index + '-acc_dev').val()
	modelAccTest = $('#' + index + '-acc_test').val()

	$('#model_description').text(modelDescription)
	$('#acc_val_train').text(modelAccTrain)
	$('#acc_val_dev').text(modelAccDev)
	$('#acc_val_test').text(modelAccTest)
}


$( document ).ready(function() {
    updateModelInfo()
    updateUnsharedOptions()
    updateFilterFields()

    // place handlers
    $('#form_model_selection').change(function(a){
		index = $('#form_model_selection')[0].selectedIndex
		$('#selected_model_index').val(index)
		updateModelInfo()
    })

    $('input[name=shared_or_not_shared]').on('change', updateUnsharedOptions)
    $('#filter_single_threshold_opt').on('change', updateFilterFields)
    $('#filter_double_threshold_opt').on('change', updateFilterFields)

    $('#btn_run').click(function(){
    	// TODO clear old output
    	console.log('jo')
    	$.ajax({
    		type: 'POST',
    		url: '/alignment_results',
    		data: $('#form').serialize(),
    		success: function(data) {
    			splitted = data.split(';')
    			$('#prediction_lbl').text(splitted[0])
                $("#result_image").attr("src","/images/" + splitted[1]);
                $('#result').show()
    		},
    		error: function(){
    			alert('Server error. Either a bug or some wrong input values.')
    		},
    		beforeSend: function(){
    			$('#prediction_lbl').text('')
                $('#result').hide()
    		}

    	});

    })
});