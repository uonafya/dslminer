import psycopg2
import click
from flask import current_app, g
from flask.cli import with_appcontext


class database:

    def get_db_con(self):
        try:

            self.connection = psycopg2.connect(user = "postgres",
                                          password = "",
                                          host = "127.0.0.1",
                                          port = "5432",
                                          database = "mohdsl")
            connection =self.connection
            self.cursor = connection.cursor()
            # Print PostgreSQL Connection properties
            print ( connection.get_dsn_parameters(),"\n")

            # Print PostgreSQL version
            self.cursor.execute("SELECT version();")
            record = self.cursor.fetchone()
            print("You are connected to - ", record,"\n")

        except (Exception, psycopg2.Error) as error :
            print ("Error while connecting to PostgreSQL", error)

        return (connection, self.cursor)


    def close_db_con(self):
        if(self.connection):
            self.cursor.close()
            self.connection.close()
            print("PostgreSQL connection is closed")
