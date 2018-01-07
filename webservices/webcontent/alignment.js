
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

function enableDisableDimensionButtons(xSize, ySize, selectedX, selectedY) {
    $('.btn_direction').prop('disabled', false);
    if(selectedX == 0) {
        $('#btn_left').prop('disabled', true);
    }
    if(selectedY == 0) {
        $('#btn_down').prop('disabled', true);
    }
    if(selectedX == xSize - 1) {
        $('#btn_right').prop('disabled', true);
    }
    if(selectedY == ySize - 1) {
        $('#btn_up').prop('disabled', true);
    }
    if ($('#destination_index_x').val() != $('#origin_index_x').val() || $('#destination_index_y').val() != $('#origin_index_y').val()) {
        $('#btn_apply').show(); 
        $('#placeholder_btn_apply').hide()
    } else {
        $('#btn_apply').hide();
        $('#placeholder_btn_apply').show()
    }
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
    initActivatingCheckboxes()
    hideAllHidden()

    $('.my-container').hide()
    $('#word-alignment-container').show()
    
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
    		url: '/~max/bla/alignment_results',
    		data: $('#form').serialize(),
    		success: function(data) {
    			splitted = data.split(';')
    			$('#prediction_lbl').text(splitted[0])
                $("#result_image").attr("src","/~max/bla/images/" + splitted[1]);
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

        function createTooltipText(matrix, allData) {
            return plotData.z.map(function(row, i) {
                return row.map(function(item, j) {
                    selectedDims = allData[i][j]

                    hintText = '<b>Premise</b> - <b>Hypothesis</b><br />';
                    dict = {};
                    for (var x=0; x< selectedDims.length; x++){
                        item = selectedDims[x]
                        key = item.p + ' - ' + item.h

                        if ( key in dict) {
                            dict[key] += 1
                        } else {
                            dict[key] = 1
                        }
                    }

                    for (key in dict) {
                        if (dict.hasOwnProperty(key)) {
                            hintText += '(' + dict[key] + ') ' + key + '<br />'
                        }
                    }

                    return hintText
                })
            })
        }

        function isInAdaptedData(indexX, indexY, adaptedData) {
            for (var i=0; i < adaptedData.length; i++) {
                if(indexY == adaptedData[i][0] && indexX == adaptedData[i][1]) {
                    return true;
                }
            }
            return false;
        }

        function removeFromAdaptedData(indexX, indexY, adaptedData) {
            var index = -1;
            for (var i=0; i < adaptedData.length; i++) {
                if(indexY == adaptedData[i][0] && indexX == adaptedData[i][1]) {
                    index = i;
                    break
                }
            }
            if (index != -1) {
                adaptedData.splice(index, 1)
            }
            return adaptedData;
        }

        function addInAdaptedData(indexX, indexY, adaptedData, val) {
            for (var i=0; i < adaptedData.length; i++) {
                if(indexY == adaptedData[i][0] && indexX == adaptedData[i][1]) {
                    adaptedData[i][2] += val
                    return adaptedData
                }
            }
            return adaptedData;
        }

        function getFromdaptedData(indexX, indexY, adaptedData) {
            for (var i=0; i < adaptedData.length; i++) {
                if(indexY == adaptedData[i][0] && indexX == adaptedData[i][1]) {
                    return adaptedData[i][2]
                }
            }
            return -1;
        }

        function getMaxMatrixValueOfAdapted(adaptedData) {
            maxVal = 0;
            for (var i=0; i < adaptedData.length; i++) {
                if(adaptedData[i][2] > maxVal) {
                    maxVal = adaptedData[i][2]
                }
            }
            return maxVal
        }

        function createLabelMatrix(matrix, adaptedData) {
            labelData = arrayClone(matrix)
            for (var i=0; i < adaptedData.length; i++) {
                labelData[adaptedData[i][0]][adaptedData[i][1]] = adaptedData[i][2]
            }
            return labelData
        }

        function labelConfusionMatrix(sizeX, sizeY, labelData, plotData) {
            annotations = []
            for (var i = 0; i< sizeY; i++) {
                for (var j=0; j< sizeX; j++) {
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

                    annotations.push(result);
                }
            }
            return annotations
        }

        function changeSelectedData(originX, destX, originY, destY, selectedData, binSize) {
            var diffX = destX - originX
            var addToX = diffX * binSize
            var diffY = originY - destY
            var addToY = diffY * binSize

            for (var i=0; i<selectedData.length; i++) {
                selectedData[i].p_rep += addToY
                selectedData[i].h_rep += addToX
            }

            return selectedData
        }

        function buildRepresentations(data) {
            repPremise = []
            repHypothesis = []
            for(var i=0; i< data.length; i++) {
                for(var j=0; j< data[i].length; j++) {
                    for(var k=0; k<data[i][j].length; k++) {
                        var item = data[i][j][k];
                        repPremise[item.dim] = item.p_rep
                        repHypothesis[item.dim] = item.h_rep
                    }
                }
            }

            return {
                repPremise: repPremise,
                repHypothesis: repHypothesis
            };
        }

        $.ajax({
            type: 'GET',
            url: '/~max/bla/alignment_general_sample',
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

                // Tooltip text
                plotData.text = createTooltipText(plotData.z, data.props.data)
                plotData.hoverinfo = 'text'

                // Correct numbers in matrix (if coloring not for high values)
                var labelData = createLabelMatrix(plotData.z, data.props.adapted)

                // Put text in cells
                plotLayout.annotations = labelConfusionMatrix(plotData.x.length, plotData.y.length, labelData, plotData)

                Plotly.newPlot('js-plot', [plotData], plotLayout)

                // Clickhandling
                $('#js-plot').unbind('plotly_click').on('plotly_click', function(d, evtData) {
                    cell = evtData.points[0]
                    all_data = data.props.data
                    label_p = cell.y
                    label_h = cell.x
                    console.log(all_data)
                    console.log(cell.pointNumber[0])
                    selected_data = all_data[cell.pointNumber[0]][cell.pointNumber[1]]
                    info = cell.text

                    // display selected info
                    $('#selected_cell_container').show()
                    $('#selected_cell_info').html(info)
                    $('#selected_p_label').text(label_p)
                    $('#selected_h_label').text(label_h)
                    $('#selected_p_label_dest').text(label_p)
                    $('#selected_h_label_dest').text(label_h)
                    $('#destination_index_x').val(cell.pointNumber[1])
                    $('#destination_index_y').val(cell.pointNumber[0])
                    $('#origin_index_x').val(cell.pointNumber[1])
                    $('#origin_index_y').val(cell.pointNumber[0])
                    enableDisableDimensionButtons(plotData.x.length, plotData.y.length, cell.pointNumber[1], cell.pointNumber[0])
                    $('.btn_direction').unbind('click').click(function() {
                        var dirX = parseInt($(this).data('dirx'))
                        var dirY = parseInt($(this).data('diry'))
                        
                        var $currentX = $('#destination_index_x')
                        var $currentY = $('#destination_index_y')
                        var nextX = parseInt($currentX.val()) + dirX
                        var nextY = parseInt($currentY.val()) + dirY
                        $currentX.val(nextX)
                        $currentY.val(nextY)

                        $('#selected_p_label_dest').text(plotData.y[nextY])
                        $('#selected_h_label_dest').text(plotData.x[nextX])
                        enableDisableDimensionButtons(plotData.x.length, plotData.y.length, nextX, nextY)
                    });

                    $('#btn_repredict').unbind('click').click(function() {
                        var dataSend = buildRepresentations(data.props.data)
                        dataSend.model = $('#form_model_selection_general').val()
                        $.ajax({
                            type: 'POST',
                            url: '/~max/bla/predict_representations',
                            data: JSON.stringify(dataSend),
                            contentType: "application/json; charset=utf-8",
                            success: function(result) {
                                console.log(result)
                                alert(result)
                                $('#prediction_lbl').text(result)
                            },
                            error: function(){
                                alert('Server error. Either a bug or some wrong input values.')
                            },
                            beforeSend: function(){
                                $('#prediction_lbl').text('')
                            }

                        });
                    })

                    $('#btn_apply').unbind('click').click(function() {
                        $('#js-plot').html('')
                        var destX = $('#destination_index_x').val()
                        var destY = $('#destination_index_y').val()
                        var originX = $('#origin_index_x').val()
                        var originY = $('#origin_index_y').val()

                        // adapt matrix
                        selectedZ = plotData.z[originY][originX]
                        plotData.z[originY][originX] = 0
                        if (isInAdaptedData(originX, originY, data.props.adapted)) {
                            selectedZ = getFromdaptedData(originX, originY, data.props.adapted)
                            data.props.adapted = removeFromAdaptedData(originX, originY, data.props.adapted)
                        }
                        
                        // adapt general data
                        if (isInAdaptedData(destX, destY, data.props.adapted)) {
                            data.props.adapted = addInAdaptedData(destX, destY, data.props.adapted, selectedZ)
                        } else {
                            plotData.z[destY][destX] += selectedZ
                            if (plotData.z[destY][destX] > data.props.maxVal) {
                                data.props.adapted = data.props.adapted.concat([[destY, destX, plotData.z[destY][destX]]])
                                plotData.z[destY][destX] = data.props.maxVal
                            }
                        }
                        selectedData = data.props.data[originY][originX]
                        selectedData = changeSelectedData(originX, destX, originY, destY, selectedData, data.props.binSize)
                        data.props.data[originY][originX] = []
                        data.props.data[destY][destX] = data.props.data[destY][destX].concat(selectedData)

                        plotData.text = createTooltipText(plotData.z, data.props.data)
                        var labelMatrix = createLabelMatrix(plotData.z, data.props.adapted)
                        plotLayout.annotations = labelConfusionMatrix(plotData.x.length, plotData.y.length, labelMatrix, plotData)
                        Plotly.newPlot('js-plot', [plotData], plotLayout)
                    })
                })
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
            url: '/~max/bla/alignment_general_results',
            data: $('#form_general').serialize(),
            success: function(data) {
                splitted = data.split(';')
                var imgEntailment = splitted[0]
                var imgContradiction = splitted[1]
                var imgNeutral = splitted[2]

                $("#general_avg_result_entailment").attr("src","/~max/bla/images/" + imgEntailment);
                $("#general_avg_result_contradiction").attr("src","/~max/bla/images/" + imgContradiction);
                $("#general_avg_result_neutral").attr("src","/~max/bla/images/" + imgNeutral);
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