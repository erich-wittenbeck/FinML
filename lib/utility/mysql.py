
import MySQLdb

def sql_compliant_str(value):

    return "'%s'" % value if type(value) is str else str(value)

def sql_bit_to_bool(value):

    if value == b'\x00':
        return False
    else:
        return True

class MySQLAccessor:

    def __init__(self, host, usr, pwd, db):
        self.__db = MySQLdb.connect(host=host, user=usr, passwd=pwd, db=db)
        self.__cur = self.__db.cursor()

    # Reading operations

    def __retrieve_from_table_row(self, table_name, uuid, column_names):

        # Cursor object

        cur = self.__cur

        # Create list of columns to be selected

        columns = ",".join(column_names)

        # Create conditional expression

        condition = "uuid=" + sql_compliant_str(uuid)

        # Create the select-statement

        statement = "SELECT %s FROM %s WHERE %s" % (columns, table_name, condition)

        # Execute

        cur.execute(statement)

        # Return result

        result = cur.fetchone()

        return result

    def __retrieve_potential_duplicates(self, table_name, unique_index_dict):

        # Cursor object

        cur = self.__cur

        # Create list of indexed columns

        # columns = "uuid, %s" % ','.join(list(unique_index_dict.keys()))

        # Create list of selectors

        selectors = " and ".join([key + '=' + sql_compliant_str(value)
                                  for key, value in unique_index_dict.items()])

        # Create the select-statement

        statement = "SELECT uuid FROM %s WHERE %s" % (table_name, selectors)

        # Execute statement

        cur.execute(statement)

        # Check for duplicates

        if cur.rowcount == 0:
            return None
        else:
            return cur.fetchone()[0]

    # Writing operations

    def __insert_into_table(self, table_name, row_dict):

        # Cursor object

        cur = self.__cur

        # Create uuid

        cur.execute("SELECT UUID()")
        uuid, = cur.fetchone()

        # Create tuples of columns and values

        columns = "(uuid, %s)" % ','.join(list(row_dict.keys()))
        values = "('%s', %s)" % (uuid, ','.join(sql_compliant_str(v) for v in list(row_dict.values())))

        # Create actual insertion-statement

        statement = "INSERT INTO %s %s VALUES %s" % (table_name, columns, values)

        # Execute statement

        cur.execute(statement)

        # Return uuid of new data-row

        return uuid

    def __update_table_row(self, table_name, uuid, update_dict):

        # Cursor object

        cur = self.__cur

        # Create list of assingments

        assignments = ",".join([key + '=' + sql_compliant_str(value)
                                for key, value in update_dict.items()])

        # Create conditional expression

        condition = "uuid=" + sql_compliant_str(uuid)

        # Create the update-statement

        statement = "UPDATE %s SET %s WHERE %s" % (table_name, assignments, condition)

        # Execute statement

        cur.execute(statement)

        return

    # Public API

    def get_foreign_keys_from_evaluation(self, model, uuid):

        table_name = model + '_evaluation'
        column_names = ['hyperparams_uuid',
                        'f1_scores_uuid',
                        'precision_scores_uuid',
                        'recall_scores_uuid']

        return self.__retrieve_from_table_row(table_name, uuid, column_names)

    def insert_scores(self, metric, pos, ntr, neg):

        table_name = metric + '_scores'
        row_dict = {'pos' : pos, 'ntr' : ntr, 'neg' : neg}

        created_uuid = self.__insert_into_table(table_name, row_dict)

        return created_uuid

    def insert_hyperparams(self, model, hyperparams):

        table_name = model + '_hyperparams'

        created_uuid = self.__insert_into_table(table_name, hyperparams)

        return created_uuid

    def insert_features(self, features_list):

        table_name = 'features'
        row_dict = {'features_list' : ';'.join(features_list)}

        duplicate_uuid = self.__retrieve_potential_duplicates(table_name, row_dict)

        if duplicate_uuid == None:
            return self.__insert_into_table(table_name, row_dict)
        else:
            return duplicate_uuid

    def insert_evaluation(self, model, metaparams, features_uuid):

        table_name = model + '_evaluation'
        row_dict = metaparams.copy()

        row_dict['features_uuid'] = features_uuid

        duplicate_uuid = self.__retrieve_potential_duplicates(table_name, row_dict)

        if duplicate_uuid == None:
            return False, self.__insert_into_table(table_name, row_dict)
        else:
            return True, duplicate_uuid

    def update_evaluation(self, model, evaluation_uuid, hyperparams_uuid, f1_scores_uuid, precision_scores_uuid, recall_scores_uuid):

        table_name = model + '_evaluation'
        update_dict = {'hyperparams_uuid' : hyperparams_uuid,
                       'f1_scores_uuid' : f1_scores_uuid,
                       'precision_scores_uuid' : precision_scores_uuid,
                       'recall_scores_uuid' : recall_scores_uuid}

        self.__update_table_row(table_name, evaluation_uuid, update_dict)

    def update_hyperparams(self, model, uuid, hyperparam_dict):

        table_name = model + '_hyperparams'

        self.__update_table_row(table_name, uuid, hyperparam_dict)

    def update_scores(self, metric, uuid, pos, ntr, neg):

        table_name = metric + '_scores'
        update_dict = {'pos' : pos,
                       'ntr' : ntr,
                       'neg' : neg}

        self.__update_table_row(table_name, uuid, update_dict)

    def commit(self):

        self.__db.commit()