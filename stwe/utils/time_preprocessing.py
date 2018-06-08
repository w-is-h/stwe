Trans = {'minute': 60,
         'hour': 60*60,
         'day': 60*60*24,
         'week': 60*60*24*7,
         'month': 60*60*24*30}

class TimeTransform(object):
    """Given a dataset iterator it builds the time_transform fucntion,
    that has the ability to normalize the time and transform it from
    seconds to minutes, hurs...
    """

    def __init__(self, time_unit='minute', normalize=True):
        """ Initialize

        time_unit:  the output time unit when running transform. The input
            time in TUW is represented in seconds. If we set /minutes/ the 
            output will be represented in minutes as an int. Options:
            'minute', 'hour', 'day', 'week', 'month'
        normalize:  should the start_time in the input data be set to zero
        """
        self.time_unit = Trans[time_unit]
        self.normalize = normalize
        self.start_time = None
        self.end_time = None


    def set_start_end_time(self, data_iterator):
        """ This function finds the start/end time in the 
        dataset, and normalizes it if needed

        data_iterator:  the dataset iterator
        """
        for tmp in data_iterator:
            start = tmp[0]
            break
        for tmp in data_iterator:
            pass
        end = tmp[0]

        self.start_time = start
        self.end_time = end


    def get_transformed_start_end(self):
        """Transforms the start/end time according 
        to given options.

        return:  (new_start_time, new_end_time)
        """

        if self.normalize:
            end = self.end_time - self.start_time
            return (0, end // time_unit)
        else:
            return (self.start_time // self.time_unit,
                    self.end_time // self.time_unit)


    def set_time_unit(self, time_unit):
        self.time_unit = Trans[time_unit]


    def transform(self, time):
        if self.normalize:
            return (time - self.start_time) // self.time_unit
        else:
            return time // self.time_unit

