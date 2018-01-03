
function arrayClone( arr ) {
    var i, copy;
    if( Array.isArray( arr ) ) {
        copy = arr.slice( 0 );
        for( i = 0; i < copy.length; i++ ) {
            copy[ i ] = arrayClone( copy[ i ] );
        }
        return copy;
    } else if( typeof arr === 'object' ) {
        throw 'Cannot clone array containing an object!';
    } else {
        return arr;
    }

}


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

function showWindow(containerId, $btn) {
    $('.my-container').hide()
    $('li.nav-item').removeClass('active')
    $btn.parent().addClass('active')
    $(containerId).show()

}

function initGeneralAlignment() {

}

function hideAllHidden() {
    $('.hidden').hide()
}

function initActivatingCheckboxes() {
    $('.activate-cb').on('change', function() {
        checked = $(this).is(':checked')

        $target = $('#' + $(this).data('activate'))
        $target.prop('disabled', !checked)
    })

    $('.activate-cb').trigger('change')

}


$( document ).ready(function() {
    updateModelInfo()
    updateUnsharedOptions()
    updateFilterFields()
    initGeneralAlignment()
    initActivatingCheckboxes()
    hideAllHidden()
    
    // place handlers
    $('#form_model_selection').change(function(a){
		index = $('#form_model_selection')[0].selectedIndex
		$('#selected_model_index').val(index)
		updateModelInfo()
    })

    // Navigation handlers
    $('#btn_word_alignment').click(function(){
        showWindow('#word-alignment-container', $(this))
    })
    $('#btn_general_alignment').click(function(){
        showWindow('#general-alignment-container', $(this))
    })


    $('input[name=shared_or_not_shared]').on('change', updateUnsharedOptions)
    $('#filter_single_threshold_opt').on('change', updateFilterFields)
    $('#filter_double_threshold_opt').on('change', updateFilterFields)

    $('#btn_run').click(function(){
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

    $('#btn_run_general_sample').click(function() {
        $.ajax({
            type: 'GET',
            url: '/alignment_general_sample',
            data: $('#form_general_sample').serialize(),
            success: function(data) {
                data = $.parseJSON(data);
                $('#general_sample_results').show()

                $('#prediction_sample_general').text(data.extra.prediction)

                plotData = data.plot
                plotData.type = 'heatmap'

                var plotLayout = {
                    annotations: [],
                    xaxis: {
                        ticks: '',
                        side: 'top',
                        tickangle: -45,
                        title: 'Hypothesis'

                    }, 
                    yaxis: {
                        ticks: '',
                        title: 'Premise'
                    }
                }

                plotData.text = plotData.z.map(function(row, i) {
                    all_data = data.props.data
                    return row.map(function(item, j) {
                        selected_dims = all_data[i][j]
                        console.log(selected_dims)

                        hintText = '<b>Premise</b> - <b>Hypothesis</b><br />';
                        for (var x=0; x< selected_dims.length; x++){
                            item = selected_dims[x]
                            hintText += item.p + '(' + (item.p_rep).toFixed(3) + ') - ' + item.h + '(' + (item.h_rep).toFixed(3) + ')<br />'
                        }

                        return hintText
                    })
                })
                plotData.hoverinfo = 'text'


                adapted = data.props.adapted
                labelData = arrayClone(plotData.z)
                for (var i=0; i < adapted.length; i++) {
                    labelData[adapted[i][0]][adapted[i][1]] = adapted[i][2]
                }

                for (var i = 0; i< plotData.y.length; i++) {
                    for (var j=0; j< plotData.x.length; j++) {
                        var val = labelData[i][j]
                        var result = {
                            xref: 'x1',
                            yref: 'y1',
                            x: plotData.x[j],
                            y: plotData.y[i],
                            text: val,
                            font: {
                                family: 'Arial',
                                size: 12,
                                color: '#000000'
                            },
                            showarrow: false
                        };

                        plotLayout.annotations.push(result);
                    }
                }



                Plotly.newPlot('js-plot', [plotData], plotLayout)

            },
            error: function(){
                alert('Server error. Either a bug or some wrong input values.')
            },
            beforeSend: function(){
                hideAllHidden()
            }
        })
    })

    $('#btn_run_general').click(function(){
        $.ajax({
            type: 'POST',
            url: '/alignment_general_results',
            data: $('#form_general').serialize(),
            success: function(data) {
                splitted = data.split(';')
                var imgEntailment = splitted[0]
                var imgContradiction = splitted[1]
                var imgNeutral = splitted[2]

                $("#general_avg_result_entailment").attr("src","/images/" + imgEntailment);
                $("#general_avg_result_contradiction").attr("src","/images/" + imgContradiction);
                $("#general_avg_result_neutral").attr("src","/images/" + imgNeutral);
                $('#general_averaged_results').show()
            },
            error: function(){
                alert('Server error. Either a bug or some wrong input values.')
            },
            beforeSend: function(){
                hideAllHidden()
            }

        });
    })
});