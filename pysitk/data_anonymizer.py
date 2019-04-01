##
# \file data_anonymizer.py
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       Dec 2016
#


# Import libraries
import string
import random
import string
import cPickle
import datetime
import os
import re

# Import modules
import pysitk.python_helper as ph


class DataAnonymizer(object):

    def __init__(self,
                 dictionary=None,
                 identifiers=None,
                 prefix_identifiers="",
                 filenames=None):

        self._dictionary = dictionary
        self._identifiers = identifiers
        self._prefix_identifiers = prefix_identifiers
        self._filenames = filenames

    ##
    # Generate identifiers
    # \date       2016-12-06 18:30:56+0000
    #
    # \param      self    The object
    # \param      length  The length
    #
    # \return     { description_of_the_return_value }
    #
    def generate_identifiers(self, randomized=False):

        if self._filenames is None:
            raise ValueError("Filenames are not set yet")

        # Create random identifier based on string
        if randomized:

            # Define amount of digits of random identifier
            digits = 4

            self._identifiers = [None] * len(self._filenames)
            for j in range(0, len(self._filenames)):
                self._identifiers[j] = ''.join(random.choice(
                    string.ascii_uppercase + string.digits)
                    for i in range(digits))

        # Identifier based on alphabet
        else:
            # ['a', 'b', 'c', ...]
            alphabet_str = list(string.ascii_lowercase)

            # Set identifiers
            self._identifiers = alphabet_str[0:len(self._filenames)]

    ##
    # Sets/Gets the identifiers.
    # \date       2016-12-06 18:29:49+0000
    #
    def set_identifiers(self, identifiers):
        self._identifiers = identifiers

    def get_identifiers(self):
        return self._identifiers

    def read_nifti_filenames_from_directory(self, directory):
        pattern = "([a-zA-Z0-9_]+)[.](nii.gz|nii)"
        p = re.compile(pattern)
        filenames = [p.match(f).group(1)
                     for f in os.listdir(directory) if p.match(f)]
        self._filenames = filenames

    ##
    # Sets/Gets filenames
    # \date       2016-12-06 18:29:59+0000
    #
    def set_filenames(self, filenames):
        self._filenames = filenames

    def get_filenames(self):
        return self._filenames

    ##
    # Set/Get the identifier prefix
    # \date       2016-12-06 18:30:19+0000
    #
    def set_prefix_identifiers(self, prefix_identifiers):
        self._prefix_identifiers = prefix_identifiers

    def get_prefix_identifiers(self):
        return self._prefix_identifiers

    ##
    # Sets/Gets dictionary
    # \date       2016-12-06 18:29:59+0000
    #
    def set_dictionary(self, dictionary):
        self._dictionary = dictionary

    def get_dictionary(self):
        return self._dictionary

    ##
    # Generate a random dictionary based on given filenames and identifiers
    # \date       2016-12-06 18:33:32+0000
    #
    # \param      self  The object
    # \post       self._dictionary created
    #
    def generate_randomized_dictionary(self):

        self._dictionary = {}

        if len(self._filenames) is not len(self._identifiers):
            raise ValueError("Length of filenames does not match identifiers")

        # Shuffle identifiers
        random.shuffle(self._identifiers)

        # Create dictionary
        for i in range(0, len(self._filenames)):
            basename = os.path.basename(os.path.basename(self._filenames[i]))
            filename, ext = ph.strip_filename_extension(basename)

            # Update identifier including the prefix
            self._identifiers[i] = "%s%s.%s" % (
                self._prefix_identifiers,
                self._identifiers[i],
                ext
            )

            # Create dictionary
            self._dictionary[self._identifiers[i]] = basename

    ##
    # Writes a dictionary.
    # \date       2016-12-06 19:26:22+0000
    #
    # \param      self             The object
    # \param      path_to_file     The path to file (".o" extension)
    # \param      filename_backup  The filename backup
    # \param      verbose          The verbose
    #
    def write_dictionary(self,
                         path_to_file,
                         filename_backup=None,
                         verbose=False):

        directory = os.path.dirname((path_to_file))
        filename, ext = ph.strip_filename_extension(
            os.path.basename(path_to_file))
        ph.create_directory(directory)

        # Write backup file (human readable)
        if filename_backup is None:
            path_to_file_backup = os.path.join(
                directory, "%s_backup_human_readable.txt" % filename)

        # Save randomized dictionary
        f = open(path_to_file, 'wb')
        cPickle.dump(self._dictionary, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()

        date = ph.get_current_date()
        time = ph.get_current_time()
        file_handle = open(path_to_file_backup, "w")
        text = "## Randomized Dictionary " + date + " " + time + "\n"
        file_handle.write(text)
        file_handle.close()

        # Print in an alphabetical order
        keys = sorted(self._dictionary.keys())
        for i in range(0, len(self._filenames)):
            file_handle = open(path_to_file_backup, "a")
            text = keys[i] + " : " + self._dictionary[keys[i]] + "\n"
            file_handle.write(text)
            file_handle.close()
            if verbose:
                print("\t%s : %s" % (keys[i], self._dictionary[keys[i]]))

        ph.print_info("Anonymization dictionary written to '%s'" %
                      path_to_file)

    ##
    # Reads a dictionary.
    # \date       2016-12-06 19:35:51+0000
    #
    # \param      self          The object
    # \param      path_to_file  The path to file
    #
    def read_dictionary(self, path_to_file):

        # Read dictionary
        f = open(path_to_file, 'rb')
        self._dictionary = cPickle.load(f)
        f.close()

        # Retrieve identifiers and filenames
        self._identifiers = self._dictionary.keys()

    ##
    # Print dictionary line by line
    # \date       2016-12-06 19:47:12+0000
    #
    # \param      self  The object
    #
    def print_dictionary(self):

        # Print in an alphabetical order
        print("Content of current dictionary:")
        keys = sorted(self._dictionary.keys())
        for i in range(0, len(self._filenames)):
            print("\t%s : %s" % (keys[i], self._dictionary[keys[i]]))

    def anonymize_files(self, dir_output):
        ph.create_directory(dir_output)

        filenames_in = [os.path.basename(f) for f in self._filenames]

        for i in range(0, len(self._filenames)):
            filename_anonymized = self._identifiers[i]
            filename_original = self._dictionary[
                self._identifiers[i]]
            try:
                index = filenames_in.index(filename_original)
            except ValueError:
                raise IOError(
                    "Given filenames (--filenames) do not match the ones given in the dictionary")

            path_to_file_anon = os.path.join(dir_output, filename_anonymized)

            cmd = "cp -p "
            cmd += self._filenames[index] + " "
            cmd += path_to_file_anon + " "
            # print(cmd)
            ph.execute_command(cmd)

    ##
    # Reveals the anonymization and adds the original filename next to the
    # encryption.
    # \date       2016-12-06 20:27:23+0000
    #
    # \param      self                The object
    # \param      directory           The directory
    # \param      filename_extension  The filename extension
    #
    # \return     revealed filenames as list of strings
    #
    def reveal_anonymized_files(self, directory):
        ph.create_directory(directory)

        filenames_revealed = []
        for i in range(0, len(self._filenames)):
            basename_anonymized = os.path.basename(self._filenames[i])
            filename_anonymized = ph.strip_filename_extension(basename_anonymized)[0]
            try:
                basename_revealed = self._dictionary[basename_anonymized]
            except KeyError:
                raise IOError("Dictionary does not match given (anonymized) filenames")
            filename_revealed = "%s_%s" % (
                filename_anonymized, basename_revealed)

            # filename_anonymized = self._identifiers[i] + filename_extension
            # filename_revealed = self._identifiers[i] + "_" + \
            #     self._dictionary[self._identifiers[i]] + filename_extension
            # filename_revealed = re.sub("_masked_srr", "", filename_revealed)

            # path_to_file_anon = os.path.join(directory, filename_anonymized)
            path_to_file_reve = os.path.join(directory, filename_revealed)

            # if not os.path.isfile(path_to_file_anon):
            #     print("%s: Nothing to reveal" % (filename_anonymized))

            cmd = "cp -p "
            cmd += self._filenames[i] + " "
            cmd += path_to_file_reve + " "
            # print(cmd)
            ph.execute_command(cmd)

            filenames_revealed.append(filename_revealed)
        return filenames_revealed

